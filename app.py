import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from embedding_loader import get_vector_store

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Validate environment variables
if not all([azure_api_key, azure_endpoint, api_version, chat_deployment]):
    st.error("Missing Azure OpenAI configuration. Please check your .env file.")
    st.stop()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=api_version,
    deployment_name=chat_deployment,
)

# Initialize embeddings (for explicit query embedding)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=api_version,
    deployment=embedding_deployment,
)

# Get vector store from embedding loader
vector_store = get_vector_store()

# Custom prompt template
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Answer as a helpful AI assistant using this information:\n{context}\nQuestion: {question}\nProvide a clear and concise response."
)

# Initialize retrieval chain
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

# Streamlit UI configuration
st.set_page_config(page_title="AI Chat Assistant", page_icon=":robot:")
st.header(" AI Chat Assistant - Municipality Plan and Population Data")
st.markdown("Ask me anything based on the pre-embedded documents!")

# Initialize conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Chat container
messages = st.container(border=True, height=600)

# Display conversation history
for entry in st.session_state.conversation:
    messages.chat_message("user").write(entry["user"])
    messages.chat_message("assistant").write(entry["assistant"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    if vector_store._collection.count() == 0:
        st.warning("No documents have been embedded yet. Please run 'embed_files.py' with files in the 'data' folder.")
    else:
        # Add user message to conversation
        st.session_state.conversation.append({"user": prompt, "assistant": ""})
        messages.chat_message("user").write(prompt)
        
        # Generate and display response with explicit query embedding
        with st.spinner("Thinking..."):
            try:
                # Explicitly embed the query (for demonstration)
                query_embedding = embeddings.embed_query(prompt)
                st.write(f"Query embedding generated (length: {len(query_embedding)})")  # Debug output
                
                # Proceed with retrieval and response generation
                response = retrieval_chain.invoke({"query": prompt})
                answer = response["result"]
                st.session_state.conversation[-1]["assistant"] = answer
                messages.chat_message("assistant").write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.conversation[-1]["assistant"] = "Sorry, I encountered an error."

# Sidebar information
st.sidebar.markdown("""
### About
This AI chat assistant uses pre-embedded documents:
- Use this chat interface
- Ask questions based on the documents
""")