from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os
import pandas as pd
import glob
import logging
from llama_index.core.llms import ChatMessage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

# Set logging configuration at the top of the file
logging.basicConfig(
    level=logging.ERROR,  # Can be DEBUG, INFO, WARNING, ERROR, or CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Add this before the VectorStoreManager class
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

# Add this before initializing the VectorStoreManager
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4-turbo-preview").encode
)
callback_manager = CallbackManager([token_counter])

class VectorStoreManager:
    MILVUS_CONFIG = {
        "uri": "http://localhost:19530",
        "dim": 3072,
    }

    def __init__(self, llm, embedding_model):
        """
        Initialize VectorStoreManager with pre-instantiated LLM and embedding models.
        
        Args:
            llm: An instantiated LLM model (e.g., OpenAI instance)
            embedding_model: An instantiated embedding model (e.g., OpenAIEmbedding instance)
        """
        self.llm = llm
        self.embed_model = embedding_model
        self._initialize_settings()

    def _initialize_settings(self):
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.callback_manager = callback_manager
        
    # IP, COSINE and L2 are the only supported metrics
    def _get_milvus_vectorstore(self, collection_identifier):
        return MilvusVectorStore(
            uri=self.MILVUS_CONFIG["uri"],
            collection_name=collection_identifier,
            dim=self.MILVUS_CONFIG["dim"],
            overwrite=False,
            similarity_metric="L2",
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
        csv_files = glob.glob("data/*.csv")

        if not csv_files:
            raise FileNotFoundError("No CSV files found in data directory")

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Convert column names to uppercase for consistency
            df.columns = df.columns.str.upper()

            for _, row in df.iterrows():
                # Combine all available text fields
                text_parts = []
                metadata = {
                    "tree_id": row["TREE_ID"],
                    "version": row["VERSION"],
                    "id": row["ID"],
                    "type": row["TYPE"],
                }

                # Optional title fields in uppercase with their prefixes
                title_fields = {
                    "CURRICULUM_TITLE": "CURRICULUM: ",
                    "UNIT_TITLE": "UNIT: ",
                    "LESSON_TITLE": "LESSON: ",
                    "ACTIVITY_TITLE": "ACTIVITY: ",
                    "ASSESSMENT_TITLE": "ASSESSMENT: ",
                    "QUESTION_TITLE": "QUESTION: ",
                }

                # Add available titles to text with their prefixes
                for field, prefix in title_fields.items():
                    if field in row and pd.notna(row[field]):
                        text_parts.append(f"{prefix}{row[field]}")
                        # Store UUID if available
                        field_name = field.split("_")[0]
                        uuid_field = f"{field_name}_UUID"
                        if uuid_field in row and pd.notna(row[uuid_field]):
                            metadata[uuid_field] = row[uuid_field]

                # Add main content
                if "BODY_VALUE" in row and pd.notna(row["BODY_VALUE"]):
                    text_parts.append(f"CONTENT: {row['BODY_VALUE']}")

                if "DESCRIPTION" in row and pd.notna(row["DESCRIPTION"]):
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
                similarity_top_k=250, 
                response_mode="tree_summarize"
            )
            
            # Extract content types first
            content_types = self.extract_content_types(query_text)
            logging.info(f"Extracted content types: {content_types}")
            
            # Get title
            title_response = query_engine.query(
                "Return ONLY the exact title of the lesson or unit mentioned in this text, "
                "without any additional commentary: " + query_text
            )
            lesson_title = title_response.response.strip()
            logging.info(f"Lesson title: {lesson_title}")
            
            # Main query
            query_response = query_engine.query(f"{query_text} titled: {lesson_title}")
            
            if not query_response or not str(query_response).strip():
                raise ValueError("Empty response received from query engine")
            
            source_matches = [
                {
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata
                }
                for node in query_response.source_nodes
            ]
            
            unique_ids = self._generate_unique_identifiers(source_matches)
            
            # Calculate costs (using current OpenAI pricing)
            gpt4_input_cost = (token_counter.prompt_llm_token_count / 1000) * 0.01  # $0.01 per 1K tokens
            gpt4_output_cost = (token_counter.completion_llm_token_count / 1000) * 0.03  # $0.03 per 1K tokens
            embedding_cost = (token_counter.total_embedding_token_count / 1000) * 0.00013  # $0.00013 per 1K tokens
            total_cost = gpt4_input_cost + gpt4_output_cost + embedding_cost

            # Print token usage and costs
            print("\nToken Usage and Costs:")
            print(f"Embedding Tokens: {token_counter.total_embedding_token_count}")
            print(f"LLM Prompt Tokens: {token_counter.prompt_llm_token_count}")
            print(f"LLM Completion Tokens: {token_counter.completion_llm_token_count}")
            print(f"Total LLM Tokens: {token_counter.total_llm_token_count}")
            print(f"\nCosts:")
            print(f"GPT-4 Input Cost: ${gpt4_input_cost:.4f}")
            print(f"GPT-4 Output Cost: ${gpt4_output_cost:.4f}")
            print(f"Embedding Cost: ${embedding_cost:.4f}")
            print(f"Total Cost: ${total_cost:.4f}\n")
            
            return query_response, source_matches, unique_ids, content_types
            
        except Exception as e:
            logging.error(f"Error in query_vector_store: {str(e)}")
            raise

    def _generate_unique_identifiers(self, matches):
        """
        Generate unique identifiers based on preference order and group by type:
        activity_uuid > assessment_uuid > lesson_uuid > unit_uuid > id
        Returns a dictionary with types as keys and lists of unique identifiers as values,
        ordered by their scores.
        """
        # Initialize dictionaries to store IDs and their scores
        grouped_ids = {
            "activity": {},
            "assessment": {},
            "lesson": {},
            "unit": {},
            "other": {},
        }

        for match in matches:
            metadata = match["metadata"]
            tree_id = metadata.get("tree_id")
            version = metadata.get("version")
            score = match["score"]

            # Check each UUID type and add to appropriate group if present
            if tree_id and version:
                if "ACTIVITY_UUID" in metadata and metadata["ACTIVITY_UUID"]:
                    unique_id = f"{tree_id}:{version}:{metadata['ACTIVITY_UUID']}"
                    grouped_ids["activity"][unique_id] = score
                elif "ASSESSMENT_UUID" in metadata and metadata["ASSESSMENT_UUID"]:
                    unique_id = f"{tree_id}:{version}:{metadata['ASSESSMENT_UUID']}"
                    grouped_ids["assessment"][unique_id] = score
                elif "LESSON_UUID" in metadata and metadata["LESSON_UUID"]:
                    unique_id = f"{tree_id}:{version}:{metadata['LESSON_UUID']}"
                    grouped_ids["lesson"][unique_id] = score
                elif "UNIT_UUID" in metadata and metadata["UNIT_UUID"]:
                    unique_id = f"{tree_id}:{version}:{metadata['UNIT_UUID']}"
                    grouped_ids["unit"][unique_id] = score
                elif "id" in metadata and metadata["id"]:
                    unique_id = f"{tree_id}:{version}:{metadata['id']}"
                    grouped_ids["other"][unique_id] = score

        # Sort each group by scores in descending order and return only the IDs
        return {
            k: [id for id, _ in sorted(v.items(), key=lambda x: x[1], reverse=True)]
            for k, v in grouped_ids.items()
        }

    def extract_content_types(self, user_prompt):
        """
        Uses OpenAI to analyze user prompt and extract requested content types.
        """
        valid_types = ['activity', 'assessment', 'lesson', 'unit']
        
        system_prompt = """
        You are a helper that identifies what types of educational content a user is asking about.
        Valid content types are: activity, assessment, lesson, and unit.
        Return ONLY a comma-separated list of the content types found, in lowercase.
        If no valid content types are found, return an empty string.
        cooldowns and warmups are also activities.
        If nothing matches, return "none".
        If user is referencing lesson or unit, it does not mean they are looking for it, analyze what user is asking for.
        Example: For "Show me all activities and assessments", return "activity,assessment"
        """
        
        response = self.llm.chat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ])
        
        # Convert response to list and validate content types
        if response.message.content:
            found_types = [t.strip() for t in response.message.content.split(',')]
            # Only return valid content types
            return [t for t in found_types if t in valid_types]
        return []

    def format_query_results(self, response, matches, unique_ids, content_types):
        """
        Format and print query results in a structured way.
        
        Args:
            response: Query response object
            matches: List of matching documents
            unique_ids: Dictionary of unique identifiers grouped by content type
            content_types: List of content types found in the query
        """
        print("-" * 20)
        print(response)
        print("-" * 20)
        print("Matches:", len(matches))
        
        # Build content type dictionary
        content_type_dict = {
            content_type: {
                'count': len(ids),
                'ids': ids
            }
            for content_type, ids in unique_ids.items()
            if ids and (content_type in content_types or content_type == 'other')
        }
        
        # Print formatted results
        print("\nContent Types Dictionary:")
        for type_key, data in content_type_dict.items():
            print(f"\n{type_key.upper()}:")
            print(f"Count: {data['count']}")
            print("IDs:")
            for id_value in data['ids']:
                print(f"  - {id_value}")
        print("-" * 20)


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Initialize models outside the VectorStoreManager
    llm = OpenAI(model="gpt-4-turbo-preview", api_key=api_key)
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=api_key,
    )

    manager = VectorStoreManager(
        llm=llm,
        embedding_model=embed_model,
    )

    # Change variable name to avoid shadowing
    TEST_COLLECTION_ID = "our_grade_8_v3"

    POPULATE_VECTOR_STORE = False

    try:
        if POPULATE_VECTOR_STORE:
            manager.populate_vector_store_cloud_storage(TEST_COLLECTION_ID)
        response, matches, unique_ids, content_types = manager.query_vector_store(
            TEST_COLLECTION_ID,
            """List all the activities in UNIT 5, return them in lesson order""",
        )
        manager.format_query_results(response, matches, unique_ids, content_types)

    except ConnectionError as e:
        print(f"Milvus connection error: {str(e)}")
    except ValueError as e:
        print(f"Data processing error: {str(e)}")
    except (FileNotFoundError, IOError) as e:
        print(f"File system error: {str(e)}")

