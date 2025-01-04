from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.readers.web import WholeSiteReader
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from typing import List, Optional, Union
import logging
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _get_milvus_config():
    user = os.getenv("MILVUS_USER")
    password = os.getenv("MILVUS_PASSWORD")
    return {
        "uri": os.getenv("MILVUS_DB_URI"),
        "token": f"{user}:{password}" if user and password else None,
    }


class VectorStoreManager:
    """
    Manages vector store operations including web crawling and querying.

    Attributes:
        MILVUS_CONFIG (dict): Configuration for Milvus vector store
        api_key (str): Google API key for LLM and embedding models
    """

    MILVUS_CONFIG = _get_milvus_config()

    def __init__(
        self, llm_model: str, embedding_model: str, api_key: Optional[str] = None
    ) -> None:
        """
        Initialize VectorStoreManager with specific models and API key.

        Args:
            llm_model: Name of the LLM model to use
            embedding_model: Name of the embedding model to use
            api_key: Optional Google API key (will try to get from env if not provided)

        Raises:
            ValueError: If GOOGLE_API_KEY is not set in environment variables
        """
        self.api_key = api_key or self._get_google_api_key()

        self.milvus_config = _get_milvus_config()
        self.milvus_dim = 768
        self._initialize_settings(llm_model, embedding_model)
        logger.info("VectorStoreManager initialized successfully")

    def _get_google_api_key(self) -> str:
        """Retrieve Google API key from environment variables."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        return api_key

    def _initialize_settings(self, llm_model: str, embedding_model: str) -> None:
        """Initialize LlamaIndex settings with specified models."""
        text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=10)

        Settings.llm = Gemini(
            model=llm_model,
            api_key=self.api_key,
        )

        Settings.embed_model = GeminiEmbedding(
            model_name=embedding_model, api_key=self.api_key
        )

        Settings.text_splitter = text_splitter
        logger.debug("Settings initialized with specified models")

    def _get_milvus_vectorstore(self, website_id=None):
        return MilvusVectorStore(
            uri=self.milvus_config["uri"],
            token=self.milvus_config["token"],
            collection_name=(
                f"website_{website_id}" if website_id else "default_collection"
            ),
            dim=self.milvus_dim,
            overwrite=False,
        )

    def _setup_chrome_options(self) -> Options:
        """Configure and return Chrome options for web crawling."""
        chrome_options = Options()

        # Basic options
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # SPA support options
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--enable-javascript")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Additional SPA options
        chrome_options.add_argument(
            "--enable-features=NetworkService,NetworkServiceInProcess"
        )
        chrome_options.add_argument(
            "--disable-features=IsolateOrigins,site-per-process"
        )
        chrome_options.add_argument("--disable-site-isolation-trials")

        return chrome_options

    def populate_vector_store_with_page(
        self, website_id: Union[str, int], base_url: str, prefix: str
    ) -> Optional[VectorStoreIndex]:
        """
        Crawl websites and populate the vector store with their content.
        """
        try:
            chrome_driver_path = ChromeDriverManager().install()
            service = Service(executable_path=chrome_driver_path, port=9515)

            driver = webdriver.Chrome(
                service=service, options=self._setup_chrome_options()
            )

            # Set various timeouts
            driver.set_page_load_timeout(30)
            driver.set_script_timeout(50)

            reader = WholeSiteReader(
                prefix=prefix, max_depth=50, uri_as_id=True, driver=driver
            )

            try:
                logger.info(f"Starting web crawl for {base_url}")
                documents = reader.load_data(base_url=base_url)
                logger.info(f"Loaded {len(documents)} documents from site crawl")
                return self._create_vector_store(website_id, documents)
            finally:
                driver.quit()

        except Exception as e:
            logger.error(
                f"Error in populate_vector_store_with_page: {str(e)}", exc_info=True
            )
            raise

    def _create_vector_store(
        self, website_id: Union[str, int], documents: List[Document]
    ) -> VectorStoreIndex:
        """Create and populate a vector store with documents."""
        if not documents:
            raise ValueError("No documents were loaded")

        vector_store = self._get_milvus_vectorstore(website_id)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
        )

        storage_context.persist()
        logger.info(f"Vector store created with {len(index.docstore.docs)} documents")
        return index

    def query_vector_store(self, website_id: Union[str, int], query_text: str) -> str:
        """
        Query the vector store with given text.

        Args:
            project_id: Project identifier
            query_text: Query string

        Returns:
            Response from the query engine

        Raises:
            Exception: If any error occurs during querying
        """
        try:
            vector_store = self._get_milvus_vectorstore(website_id)
            index = VectorStoreIndex.from_vector_store(vector_store)

            system_prompt = """You are an AI assistant chatbot that responds to user queries. 
            When responding:
            1. Be concise and direct
            2. Format lists with bullet points
            3. Only include information that is explicitly mentioned in the source documents
            4. Exclude cookie related information

            Respond based on the context provided.
            
            """

            query_engine = index.as_query_engine(
                similarity_top_k=15,
                response_mode="tree_summarize",
                streaming=True,
                system_prompt=system_prompt,
            )

            response = query_engine.query(query_text)

            if not response:
                logger.warning("Empty response received from query engine")

            return response

        except Exception as e:
            logger.error(f"Error in query_vector_store: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    try:
        MODEL_CONFIG = {
            "llm_model": "models/gemini-1.5-flash",
            "embedding_model": "models/text-embedding-004",
        }

        manager = VectorStoreManager(
            llm_model=MODEL_CONFIG["llm_model"],
            embedding_model=MODEL_CONFIG["embedding_model"],
        )

        website_id = "itv"
        logger.info(f"Starting process for Project ID: {website_id}")

        url = "https://www.itverticals.com"

        manager.populate_vector_store_with_page(website_id, url, url)

        response = manager.query_vector_store(
            website_id,
            "What does it say about optimizing google profile?",
        )

        print(response)

    except Exception as e:
        logger.error(f"Main execution error: {str(e)}", exc_info=True)
        raise
