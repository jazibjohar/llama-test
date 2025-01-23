from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os
import uuid


class VectorStoreManager:
    MILVUS_CONFIG = {"uri": "http://localhost:19530", "dim": 3072}

    def __init__(self, llm_model, embedding_model, api_key=None):
        """
        Initialize VectorStoreManager with specific models and API key.
        """
        self.api_key = api_key or self._get_openai_api_key()
        self._initialize_settings(llm_model, embedding_model)

    def _get_openai_api_key(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return api_key

    def _initialize_settings(self, llm_model, embedding_model):
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=10)

        Settings.llm = OpenAI(
            model=llm_model,
            api_key=self.api_key,
        )

        Settings.embed_model = OpenAIEmbedding(
            model_name=embedding_model, api_key=self.api_key
        )

        Settings.text_splitter = text_splitter

    def _format_collection_name(self, collection_identifier):
        """
        Formats collection_identifier into a valid Milvus collection name.
        - Replaces hyphens with underscores
        - Ensures the name starts with a letter
        - Removes any invalid characters
        """
        print(collection_identifier)
        # Convert UUID to string if needed
        collection_identifier = str(collection_identifier)
        # Replace hyphens with underscores and remove any other invalid characters
        formatted_id = collection_identifier.replace("-", "_")
        print(f"Formatted ID: {formatted_id}")
        # Ensure name starts with "collection_" to guarantee it begins with a letter
        return f"collection_{formatted_id}"

    def _get_milvus_vectorstore(self, collection_identifier):
        return MilvusVectorStore(
            uri=self.MILVUS_CONFIG["uri"],
            collection_name=self._format_collection_name(collection_identifier),
            dim=self.MILVUS_CONFIG["dim"],
            overwrite=False,
        )

    def populate_vector_store_local(self, collection_identifier):
        """Populates vector store from local directory."""
        try:
            documents = SimpleDirectoryReader("data", recursive=True).load_data()
            return self._create_vector_store(collection_identifier, documents)
        except (FileNotFoundError, IOError) as e:
            print(f"Error reading files: {str(e)}")
            raise
        except ValueError as e:
            print(f"Error processing documents: {str(e)}")
            raise

    def populate_vector_store_cloud_storage(self, collection_identifier):
        """Populates vector store from local folder."""
        try:
            documents = SimpleDirectoryReader(
                input_dir="data", recursive=True
            ).load_data()

            print(f"\nLoaded {len(documents)} documents")
            return self._create_vector_store(collection_identifier, documents)
        except (FileNotFoundError, IOError) as e:
            print(f"Error reading files: {str(e)}")
            raise
        except ValueError as e:
            print(f"Error processing documents: {str(e)}")
            raise

    def _create_vector_store(self, collection_identifier, documents):
        """Helper method for vector store creation logic."""
        if not documents:
            raise ValueError("No documents were loaded")

        print(f"Found {len(documents)} documents")
        vector_store = self._get_milvus_vectorstore(collection_identifier)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            vector_store=vector_store,
            show_progress=True,
            debug=False,
        )

        print(f"Index created with {len(index.docstore.docs)} documents")
        storage_context.persist()
        return index

    def query_vector_store(self, collection_identifier, query_text):
        try:
            vector_store = self._get_milvus_vectorstore(collection_identifier)
            index = VectorStoreIndex.from_vector_store(vector_store)

            query_engine = index.as_query_engine(
                similarity_top_k=5, response_mode="tree_summarize"
            )

            query_response = query_engine.query(query_text)

            if not query_response or not str(query_response).strip():
                print("Warning: Empty response received from query engine")

            return query_response
        except (ConnectionError, ValueError) as e:
            print(f"Error querying vector store: {str(e)}")
            raise



if __name__ == "__main__":
    # Example usage with OpenAI models
    MODEL_CONFIG = {
        "llm_model": "gpt-4-turbo-preview",
        "embedding_model": "text-embedding-3-large",
    }

    manager = VectorStoreManager(
        llm_model=MODEL_CONFIG["llm_model"],
        embedding_model=MODEL_CONFIG["embedding_model"],
    )

    # Change variable name to avoid shadowing
    test_collection_id = "my_collection"
    print(f"Collection Identifier: {test_collection_id}")
    
    populate_vector_store = False

    try:
        if populate_vector_store:
            manager.populate_vector_store_cloud_storage(test_collection_id)
        response = manager.query_vector_store(
            test_collection_id,
            "Summarize the document",
        )

        print(response)
    except ConnectionError as e:
        print(f"Milvus connection error: {str(e)}")
    except ValueError as e:
        print(f"Data processing error: {str(e)}")
    except (FileNotFoundError, IOError) as e:
        print(f"File system error: {str(e)}")
