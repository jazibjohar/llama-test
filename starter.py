from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore

# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

import os

MILVUS_CONFIG = {"uri": "http://localhost:19530", "dim": 768}

MODEL_CONFIG = {
    "llm_model": "models/gemini-1.5-flash",
    "embedding_model": "models/text-embedding-004",
}


def get_google_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    return api_key


def get_milvus_vectorstore(project_id):
    return MilvusVectorStore(
        uri=MILVUS_CONFIG["uri"],
        collection_name=f"document_vectors_{project_id}",
        dim=MILVUS_CONFIG["dim"],
    )


def populate_vector_store(project_id):
    # Setup components
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=10)

    api_key = get_google_api_key()

    # Configure settings
    Settings.llm = Gemini(
        model=MODEL_CONFIG["llm_model"],
        api_key=api_key,
    )

    Settings.embed_model = GeminiEmbedding(
        model_name=MODEL_CONFIG["embedding_model"], api_key=api_key
    )

    Settings.text_splitter = text_splitter

    # Load and index documents
    documents = SimpleDirectoryReader("data", recursive=True).load_data()
    vector_store = get_milvus_vectorstore(project_id)
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    return index


def query_vector_store(project_id, query_text):
    vector_store = get_milvus_vectorstore(project_id)
    # Create index with the existing vector store
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response


def sync_vector_store(project_id):
    vector_store = populate_vector_store(project_id)
    # Add your synchronization logic here
    pass


if __name__ == "__main__":
    # Example usage
    project_id = "demo_project_01"
    index = populate_vector_store(project_id)
    response = query_vector_store(
        project_id, "What did James Turner say about the migration project?"
    )
    print(response)
