from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

import os


text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)


documents = SimpleDirectoryReader("data", recursive=True).load_data()


# Settings.llm = Ollama(
#     model="llama3.2:latest", request_timeout=60.0, base_url="http://10.0.50.26:11434"
# )
Settings.llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)


model_name = "models/text-embedding-004"

Settings.embed_model = GeminiEmbedding(
    model_name=model_name, api_key=os.getenv("GOOGLE_API_KEY")
)

Settings.text_splitter = text_splitter


index = VectorStoreIndex.from_documents(documents)

# http://10.0.50.26:11434/api/chat

query_engine = index.as_query_engine()
response = query_engine.query(
    """What is the name of Senior DevOps Engineer? And what they said in the meeting"""
)
print(response)
