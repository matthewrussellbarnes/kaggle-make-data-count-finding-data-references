import os
import re
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import pypdf
from bs4 import BeautifulSoup
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Langfuse integration
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    )
langfuse_callback_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    )

# --- 1. Data Ingestion ---
def get_document_paths(data_dir="train"):
    pdf_dir = os.path.join(data_dir, "PDF")
    xml_dir = os.path.join(data_dir, "XML")

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    # Create a dictionary to match pdf and xml files by their base names
    file_map = {}
    for pdf_file in pdf_files:
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]
        file_map[base_name] = {'pdf': pdf_file}

    for xml_file in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        if base_name in file_map:
            file_map[base_name]['xml'] = xml_file
            
    return list(file_map.values())

def parse_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF {os.path.basename(file_path)}: {e}")
        return ""

def parse_xml(file_path):
    """Extracts text content from an XML file, preserving some structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')
        
        # Extract text from specific tags, you might need to adjust these
        title = soup.find('title')
        abstract = soup.find('abstract')
        
        body_text = ""
        for p in soup.find_all('p'):
            body_text += p.get_text(separator=' ', strip=True) + "\n"
            
        return {
            "title": title.get_text(strip=True) if title else "No Title",
            "abstract": abstract.get_text(strip=True) if abstract else "No Abstract",
            "body": body_text,
        }
    except Exception as e:
        print(f"Error reading XML {os.path.basename(file_path)}: {e}")
        return {}

def load_documents(file_paths):
    file_paths = file_paths[:1]
    docs = []
    for paths in tqdm(file_paths, desc="Loading documents"):
        pdf_path = paths.get('pdf')
        xml_path = paths.get('xml')
        doc = {
            "id": os.path.splitext(os.path.basename(pdf_path or xml_path))[0],
            "content": ""
        }
        
        if pdf_path:
            doc["pdf_content"] = parse_pdf(pdf_path)
        
        if xml_path:
            xml_data = parse_xml(xml_path)
            doc["xml_content"] = f"\nTitle: {xml_data.get('title')}\nAbstract: {xml_data.get('abstract')}\nBody:\n{xml_data.get('body')}"

        docs.append(doc)
    return docs

# --- 2. LangGraph Agent Definition ---

class AgentState(TypedDict):
    document_content: str
    data_references: List[str]
    document_id: str

# Define the nodes for the graph
def identify_data_references_node(state: AgentState):
    """
    Identifies datasets, repositories, and data mentions in the document.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at identifying references to datasets, data repositories, or mentions of data usage in scientific papers. Your task is to extract any phrases or sentences that indicate the use of data. Look for names of datasets, URLs to data repositories, or descriptions of data collection and usage."),
        ("user", "Please analyze the following document content and extract all data references:\n\n---\n\n{document_content}")
    ])
    
    llm = ChatOllama(model="llama3", temperature=0)
    
    chain = prompt | llm | StrOutputParser()
    
    data_references_text = chain.invoke(
        {"document_content": state["document_content"]},
        config={"callbacks": [langfuse_callback_handler]}
    )
    
    # Simple regex to split the response into a list of references.
    # This might need refinement based on the LLM's output format.
    references = re.split(r'\n\s*\d+\.\s*', data_references_text)
    cleaned_references = [ref.strip() for ref in references if ref.strip()]
    
    return {"data_references": cleaned_references}

# Define the graph
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("identify_data_references", identify_data_references_node)
    workflow.set_entry_point("identify_data_references")
    workflow.add_edge("identify_data_references", END)
    return workflow.compile()

# --- 3. Main Execution Logic ---
def main():
    print("Starting document processing...")
    
    # 1. Load data
    document_paths = get_document_paths()
    documents = load_documents(document_paths)
    
    # 2. Build the agent graph
    app = build_graph()
    
    # 3. Process each document
    all_results = []
    for doc in documents:
        inputs = {
            "document_content": f"PDF Content:\n{doc['pdf_content']}\n\nXML Content:\n{doc['xml_content']}",
            "document_id": doc["id"]
        }
        
        result = app.invoke(inputs, config={"callbacks": [langfuse_callback_handler]})
        
        print(f"\n--- Document ID: {doc['id']} ---")
        if result.get("data_references"):
            print("Found Data References:")
            for ref in result["data_references"]:
                print(f"- {ref}")
        else:
            print("No data references found.")

        all_results.append({
            "id": doc["id"],
            "references": result.get("data_references", [])
        })

    print("\nProcessing complete.")
    langfuse.flush() # Ensure all traces are sent

if __name__ == "__main__":
    main() 