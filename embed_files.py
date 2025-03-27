import os
from dotenv import load_dotenv
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
import pandas as pd
import pypdf
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=api_version,
    deployment=embedding_deployment,
)

# Constants
chunk_size = 500
chunk_overlap = 50
data_folder = "./data"

# Initialize vector store
vector_store = Chroma(
    collection_name="file_docs",
    embedding_function=embeddings,
    persist_directory="./db/file_chroma_db",
)

def ingest_pdf(file_path, existing_sources):
    """Ingest a PDF file if not already embedded"""
    filename = os.path.basename(file_path)
    if filename in existing_sources:
        print(f"Skipping {file_path} - already embedded")
        return
    try:
        pdf_reader = pypdf.PdfReader(file_path)
        num_pages = len(pdf_reader.pages)
        documents = []
        
        with tqdm(total=num_pages, desc=f"Embedding PDF: {filename}", unit="page") as pbar:
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text() or ""
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": page_num + 1}
                ))
                pbar.update(1)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(documents=chunks, ids=uuids)
        print(f"Embedded {len(chunks)} chunks from PDF: {file_path}")
    except Exception as e:
        print(f"Error embedding PDF {file_path}: {e}")

def ingest_excel(file_path, existing_sources):
    """Ingest an Excel file if not already embedded"""
    filename = os.path.basename(file_path)
    if filename in existing_sources:
        print(f"Skipping {file_path} - already embedded")
        return
    try:
        xl = pd.ExcelFile(file_path)
        num_sheets = len(xl.sheet_names)
        documents = []
        
        with tqdm(total=num_sheets, desc=f"Embedding Excel: {filename}", unit="sheet") as pbar:
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                headers = " | ".join(str(col) for col in df.columns)
                content = [headers]
                for _, row in df.iterrows():
                    row_str = " | ".join(str(val) for val in row)
                    content.append(row_str)
                sheet_content = "\n".join(content)
                documents.append(Document(
                    page_content=sheet_content,
                    metadata={"source": filename, "sheet": sheet_name}
                ))
                pbar.update(1)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(documents=chunks, ids=uuids)
        print(f"Embedded {len(chunks)} chunks from Excel: {file_path}")
    except Exception as e:
        print(f"Error embedding Excel {file_path}: {e}")

def embed_all_files():
    """Embed only new PDF and Excel files from the data folder with progress tracking"""
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created data folder: {data_folder}")
        print("Please place PDF and Excel files in the 'data' folder and run this script again.")
        return

    # Count existing embeddings and get existing sources
    existing_count = vector_store._collection.count()
    existing_data = vector_store._collection.get(include=["metadatas"])
    
    # Handle case where metadatas is None or empty
    if existing_data["metadatas"] is None or not existing_data["metadatas"]:
        existing_sources = set()
    else:
        existing_sources = {doc["source"] for doc in existing_data["metadatas"] if "source" in doc}
    
    print(f"Starting with {existing_count} existing embeddings from {len(existing_sources)} files")

    # Get list of files to process
    files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.pdf', '.xlsx', '.xls'))]
    if not files:
        print("No PDF or Excel files found in the data folder.")
        return

    # Filter out already embedded files
    new_files = [f for f in files if f not in existing_sources]
    if not new_files:
        print("No new files to embed.")
        return

    # Progress bar for new files
    with tqdm(total=len(new_files), desc="Processing new files", unit="file") as pbar:
        for filename in new_files:
            file_path = os.path.join(data_folder, filename)
            if filename.lower().endswith('.pdf'):
                ingest_pdf(file_path, existing_sources)
            elif filename.lower().endswith(('.xlsx', '.xls')):
                ingest_excel(file_path, existing_sources)
            pbar.update(1)
    
    new_count = vector_store._collection.count()
    print(f"File embedding completed. Total embeddings: {new_count} (Added {new_count - existing_count})")

if __name__ == "__main__":
    embed_all_files()