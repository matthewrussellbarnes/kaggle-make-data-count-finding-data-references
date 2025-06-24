import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
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

langfuse_callback_handler = CallbackHandler()

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

class DataReference(BaseModel):
    """A single data reference found in a document."""
    referenced_data: str = Field(description="The piece of text that references a dataset, data repository, or data usage.")
    citation_name: Optional[str] = Field(description="The name of the citation or reference ID, if available.", default=None)

class DataReferences(BaseModel):
    """A list of data references found in the document."""
    references: List[DataReference]

class AgentState(TypedDict):
    document_content: str
    extracted_references_text: str
    data_references: List[DataReference]
    document_id: str

# Define the nodes for the graph
def extract_data_references_node(state: AgentState):
    """
    Step 1: Extract data references as plain text from the document.
    """
    llm = ChatOllama(model="llama3", temperature=0.0)
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You must find ONLY sentences about DATA COLLECTION, DATASET USAGE, or DATA REPOSITORIES. 

CRITICAL RULE: If a sentence contains "et al." it is an AUTHOR CITATION - DO NOT EXTRACT IT.

EXTRACT sentences that describe:
✅ Data collection: "We collected...", "Data were gathered...", "Samples were obtained..."
✅ Dataset usage: "We used the [dataset name]", "Data from [database/repository]"
✅ Database access: "Downloaded from...", "Retrieved from...", "Accessed via..."  
✅ Repository mentions: "Available at [URL]", "Deposited in [repository]"
✅ Survey/experimental data: "Participants completed...", "Measurements were taken..."

NEVER EXTRACT (these are NOT data references):
❌ Author citations: "[Author] et al. (year)" or "(Author, year)"
❌ Funding organizations: grant agencies, foundations
❌ Universities/institutions: research institutions, universities
❌ Method references: "following [Author's] protocol"
❌ General study references without data mention

IMPORTANT: Focus on sentences that describe the ACTUAL COLLECTION, ACCESS, or USE of datasets/data, not citations to other research.

List each data reference on a separate line. If none found, respond with "NONE"."""),
        ("user", "Find sentences about DATA COLLECTION, DATASETS, or DATA REPOSITORIES. Skip author citations, funding, institutions.\n\nDocument:\n{document_content}")
    ])
    
    extraction_chain = extraction_prompt | llm
    
    extraction_result = extraction_chain.invoke(
        {"document_content": state["document_content"]},
        config={"callbacks": [langfuse_callback_handler]}
    )
    
    extracted_text = extraction_result.content.strip()
    
    return {"extracted_references_text": extracted_text}

def structure_data_references_node(state: AgentState):
    """
    Step 2: Structure the extracted references into DataReference objects.
    """
    extracted_text = state["extracted_references_text"]
    
    if extracted_text.upper() == "NONE" or not extracted_text:
        return {"data_references": []}
    
    llm = ChatOllama(model="llama3", temperature=0.0)
    
    structuring_prompt = ChatPromptTemplate.from_messages([
        ("system", """Convert these data references into structured format. For each reference, create a DataReference object with:
- referenced_data: the complete sentence
- citation_name: null (unless there's a specific dataset citation)"""),
        ("user", "Convert these data references to structured format:\n\n{extracted_references}")
    ])
    
    structured_llm = llm.with_structured_output(DataReferences)
    structuring_chain = structuring_prompt | structured_llm
    
    structuring_result = structuring_chain.invoke(
        {"extracted_references": extracted_text},
        config={"callbacks": [langfuse_callback_handler]}
    )
    
    return {"data_references": structuring_result.references}

# Define the graph
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("extract_data_references", extract_data_references_node)
    workflow.add_node("structure_data_references", structure_data_references_node)
    
    workflow.set_entry_point("extract_data_references")
    workflow.add_edge("extract_data_references", "structure_data_references")
    workflow.add_edge("structure_data_references", END)
    
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