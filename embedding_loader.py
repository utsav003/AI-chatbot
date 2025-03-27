import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

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

# Initialize vector store
vector_store = Chroma(
    collection_name="file_docs",
    embedding_function=embeddings,
    persist_directory="./db/file_chroma_db",
)

def get_vector_store():
    return vector_store