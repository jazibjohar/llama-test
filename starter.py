from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.readers.gcs import GCSReader
import os


class VectorStoreManager:
    MILVUS_CONFIG = {"uri": "http://localhost:19530", "dim": 768}

    def __init__(self, llm_model, embedding_model, api_key=None):
        """
        Initialize VectorStoreManager with specific models and API key.
        """
        self.api_key = api_key or self._get_google_api_key()
        self._initialize_settings(llm_model, embedding_model)

    def _get_google_api_key(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        return api_key

    def _initialize_settings(self, llm_model, embedding_model):
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=10)

        Settings.llm = Gemini(
            model=llm_model,
            api_key=self.api_key,
        )

        Settings.embed_model = GeminiEmbedding(
            model_name=embedding_model, api_key=self.api_key
        )

        Settings.text_splitter = text_splitter

    def _format_collection_name(self, project_id):
        """
        Formats project_id into a valid Milvus collection name.
        - Replaces hyphens with underscores
        - Ensures the name starts with a letter
        - Removes any invalid characters
        """
        # Convert UUID to string if needed
        project_id = str(project_id)
        # Replace hyphens with underscores and remove any other invalid characters
        formatted_id = project_id.replace("-", "_")
        # Ensure name starts with "collection_" to guarantee it begins with a letter
        return f"collection_{formatted_id}"

    def _get_milvus_vectorstore(self, project_id):
        return MilvusVectorStore(
            uri=self.MILVUS_CONFIG["uri"],
            collection_name=self._format_collection_name(project_id),
            dim=self.MILVUS_CONFIG["dim"],
            overwrite=False,
        )

    def populate_vector_store_local(self, project_id):
        """Populates vector store from local directory."""
        try:
            documents = SimpleDirectoryReader("data", recursive=True).load_data()
            return self._create_vector_store(project_id, documents)
        except Exception as e:
            print(f"Error in populate_vector_store_local: {str(e)}")
            raise

    def populate_vector_store_cloud_storage(self, project_id, bucket_name):
        """Populates vector store from Google Cloud Storage bucket folder."""
        try:
            reader = GCSReader(
                bucket=bucket_name,
                prefix=project_id+"/",
                service_account_key_path="./service-account-key.json",
                recursive=True,
                file_metadata=lambda x: {}
            )   
            resources = reader.list_resources()         
            for resource in resources:
                print(resource)

            documents = reader.load_data()
            print(f"\nLoaded {len(documents)} documents")
            return self._create_vector_store(project_id, documents)
        except Exception as e:
            print(f"Error in populate_vector_store_cloud_storage: {str(e)}")
            raise

    def _create_vector_store(self, project_id, documents):
        """Helper method for vector store creation logic."""
        if not documents:
            raise ValueError("No documents were loaded")

        print(f"Found {len(documents)} documents")
        vector_store = self._get_milvus_vectorstore(project_id)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            vector_store=vector_store,
            show_progress=False,
            debug=False,
        )

        print(f"Index created with {len(index.docstore.docs)} documents")
        storage_context.persist()
        return index

    def query_vector_store(self, project_id, query_text):
        try:
            vector_store = self._get_milvus_vectorstore(project_id)
            index = VectorStoreIndex.from_vector_store(vector_store)

            query_engine = index.as_query_engine(
                similarity_top_k=5, response_mode="tree_summarize"
            )

            response = query_engine.query(query_text)

            if not response or not str(response).strip():
                print("Warning: Empty response received from query engine")

            return response
        except Exception as e:
            print(f"Error in query_vector_store: {str(e)}")
            raise

    def sync_vector_store(self, project_id):
        vector_store = self.populate_vector_store(project_id)
        # Add your synchronization logic here
        pass


if __name__ == "__main__":
    # Example usage
    MODEL_CONFIG = {
        "llm_model": "models/gemini-1.5-flash",
        "embedding_model": "models/text-embedding-004",
    }

    manager = VectorStoreManager(
        llm_model=MODEL_CONFIG["llm_model"],
        embedding_model=MODEL_CONFIG["embedding_model"],
    )

    project_id = "490694d5-f6ba-4dc0-b523-0c73ccd75df3"
    print(f"Project ID: {project_id}")
    # Add debug prints
    try:
        manager.populate_vector_store_cloud_storage(
            project_id, "rag_test_bucket_baligh"
        )
        response = manager.query_vector_store(
            project_id,
            "List all blockers and action items from standup happened on December 26, 2024",
        )

        print(response)
    except Exception as e:
        print(f"Main execution error: {str(e)}")
