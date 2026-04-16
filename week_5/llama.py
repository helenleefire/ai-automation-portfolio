
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.anthropic import Anthropic

load_dotenv()


urls = [
  "https://docs.oracle.com/javase/8/docs/api/java/util/HashMap.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/Hashtable.html"
]

api_key=os.getenv("ANTHROPIC_API_KEY")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
Settings.llm = Anthropic(model="claude-sonnet-4-6", api_key=api_key)

documents = SimpleWebPageReader().load_data(urls)
nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine()
response = query_engine.query("What is hash map")

print(response)