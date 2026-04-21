
import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
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
documents += SimpleDirectoryReader(input_dir="./week_5/sample_data").load_data()
nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)


chat_engine = index.as_chat_engine(similarity_top_k=5, chat_mode="condense_plus_context", response_mode="tree_summarize")

def main() -> None:
  print("What do you want to ask the agent? Enter stop to end session")
  while True:
    question = input("Your question: ").strip()
    if question.lower() == "stop":
      break
    answer = chat_engine.chat(question)
    print(f'source node score: {answer.source_nodes[-1].score}')
    print(f'{answer} \n What else would you like to ask? Enter stop to end session.')

main()