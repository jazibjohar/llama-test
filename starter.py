from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os
import uuid
import pandas as pd
import glob
import logging

# Set logging configuration at the top of the file
logging.basicConfig(
    level=logging.ERROR,  # Can be DEBUG, INFO, WARNING, ERROR, or CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add this before the VectorStoreManager class
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

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
        """Populates vector store from local CSV files."""
        try:
            documents = self._load_csv_documents()
            return self._create_vector_store(collection_identifier, documents)
        except (FileNotFoundError, IOError) as e:
            print(f"Error reading files: {str(e)}")
            raise
        except ValueError as e:
            print(f"Error processing documents: {str(e)}")
            raise

    def populate_vector_store_cloud_storage(self, collection_identifier):
        """Populates vector store from CSV files."""
        try:
            documents = self._load_csv_documents()
            print(f"\nLoaded {len(documents)} documents")
            return self._create_vector_store(collection_identifier, documents)
        except (FileNotFoundError, IOError) as e:
            print(f"Error reading files: {str(e)}")
            raise
        except ValueError as e:
            print(f"Error processing documents: {str(e)}")
            raise

    def _load_csv_documents(self):
        """Load and process CSV files into LlamaIndex documents."""
        documents = []
        csv_files = glob.glob('data/*.csv')
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data directory")

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Convert column names to uppercase for consistency
            df.columns = df.columns.str.upper()
            
            for _, row in df.iterrows():
                # Combine all available text fields
                text_parts = []
                metadata = {}
                
                # Optional title fields in uppercase with their prefixes
                title_fields = {
                    'CURRICULUM_TITLE': 'CURRICULUM: ',
                    'UNIT_TITLE': 'UNIT: ',
                    'LESSON_TITLE': 'LESSON: ',
                    'ACTIVITY_TITLE': 'ACTIVITY: ',
                    'ASSESSMENT_TITLE': 'ASSESSMENT: ',
                    'QUESTION_TITLE': 'QUESTION: '
                }
                
                # Add available titles to text with their prefixes
                for field, prefix in title_fields.items():
                    if field in row and pd.notna(row[field]):
                        text_parts.append(f"{prefix}{row[field]}")
                        # Store UUID if available
                        uuid_field = f"{field}_UUID"
                        if uuid_field in row and pd.notna(row[uuid_field]):
                            metadata[uuid_field] = row[uuid_field]

                # Add main content
                if 'BODY_VALUE' in row and pd.notna(row['BODY_VALUE']):
                    text_parts.append(f"CONTENT: {row['BODY_VALUE']}")
                
                if 'DESCRIPTION' in row and pd.notna(row['DESCRIPTION']):
                    text_parts.append(f"DESCRIPTION: {row['DESCRIPTION']}")

                # Create the final text by joining all parts
                text = "\n".join(text_parts)
                
                if text.strip():  # Only create document if there's content
                    doc = Document(text=text, metadata=metadata)
                    documents.append(doc)

        return documents

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
                similarity_top_k=1000, response_mode="tree_summarize"
            )

            query_response = query_engine.query(query_text)

            if not query_response or not str(query_response).strip():
                print("Warning: Empty response received from query engine")
                
            # Extract and return the matching documents
            source_matches = []
            for node in query_response.source_nodes:
                source_matches.append({
                    'text': node.text,
                    'score': node.score,
                    'metadata': node.metadata
                })

            return query_response, source_matches
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
    test_collection_id = "our_grade_8_v2"
    print(f"Collection Identifier: {test_collection_id}")
    
    populate_vector_store = False

    try:
        if populate_vector_store:
            manager.populate_vector_store_cloud_storage(test_collection_id)
        response, matches = manager.query_vector_store(
            test_collection_id,
            "List all the activities in lesson 3 of unit 5",
        )

        print("-"*20)
        print(response)
        print("-"*20)
        print("Matches:",len(matches))
        # for match in matches:
        #     print(match['text'])
        #     print(match['metadata'])
        #     print("-"*20)
            
            
    except ConnectionError as e:
        print(f"Milvus connection error: {str(e)}")
    except ValueError as e:
        print(f"Data processing error: {str(e)}")
    except (FileNotFoundError, IOError) as e:
        print(f"File system error: {str(e)}")
