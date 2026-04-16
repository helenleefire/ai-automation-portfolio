
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.anthropic import Anthropic

load_dotenv()
api_key=os.getenv("ANTHROPIC_API_KEY")

urls = [
  "https://docs.oracle.com/javase/8/docs/api/java/util/HashMap.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/Hashtable.html"
]

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
Settings.llm = Anthropic(
    model="claude-sonnet-4-6", 
    api_key=api_key,
    system_prompt="You are a helpful agent that answers technical questions in concise manner, easy enough for non technical people to understand."
    )

documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)


query_engine = index.as_query_engine()

def main() -> None:
  print("What do you want to ask the agent? Enter stop to end session")
  while True:
    question = input("Your question: ").strip()
    if question.lower() == "stop":
      break
    answer = query_engine.query(question)
    print(f'source node score: {answer.source_nodes[-1].score}')
    print(f'{answer} \n What else would you like to ask? Enter stop to end session.')

main()