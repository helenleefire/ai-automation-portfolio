
import os
import asyncio
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.anthropic import Anthropic
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool

load_dotenv()
api_key=os.getenv("ANTHROPIC_API_KEY")
urls = [
  "https://docs.oracle.com/javase/8/docs/api/java/util/HashMap.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/Hashtable.html"
]

llm = Anthropic(
    model="claude-sonnet-4-6", 
    api_key=api_key)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
documents += SimpleDirectoryReader(input_dir="./week_5/sample_data").load_data()
nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
query_engine_tool = QueryEngineTool.from_defaults(
  query_engine=index.as_query_engine(),
  description="Used to answer questions about data structures, especially with Java"
)

agent = FunctionAgent(
  tools=[query_engine_tool],
  llm = llm,
  system_prompt="You are a helpful agent that queries given documents at all times. Mention you are using your knowledge base when refrencing them and say that you're not when you're not",
)

async def main() -> None:
  print("What do you want to ask the agent? Enter stop to end session")
  while True:
    question = input("Your question: ").strip()
    if question.lower() == "stop":
      break
    answer = await agent.run(question)
    print(f'{str(answer)} \n What else would you like to ask? Enter stop to end session.')

asyncio.run(main())