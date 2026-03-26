import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["USER_AGENT"] = "my-app/1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

url = [
  "https://docs.oracle.com/javase/8/docs/api/java/util/HashMap.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/Hashtable.html"
  ]
web_data = WebBaseLoader(url).load()

document_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
data = document_splitter.split_documents(web_data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma (
  collection_name="my_collection",
  embedding_function=embeddings,
  persist_directory="./chroma_langchain_db"
)

vectorstore.add_documents(data)

@tool
def data_retriever(query: str) -> str:
  """Reterieve relevent informtaion from existing documents to answer questions.
  Let the user know when using them."""
  search = vectorstore.similarity_search(query)
  response = ""
  if not search :
    return "I don't see relevant information in my knowledge base"
  return response.join("\n".join([doc.page_content for doc in search]))

model = init_chat_model("claude-sonnet-4-6")

checkpointer = InMemorySaver()

config = {"configurable": {"thread_id": "rag-session-1"}}

agent = create_agent(
  model=model,
  tools=[data_retriever],
  checkpointer=checkpointer,
  system_prompt="""Use the data retrieval tool when answering questions.
  Keep the response to at most a paragraph unless user asks for more detail"""
)

query1 = "What is a hashmap"
result1 = agent.invoke(
  {"messages": [{"role": "user", "content": query1}]},
  config=config
)
print(result1["messages"][-1].content)


query2 = "tell me more"
result2 = agent.invoke(
  {"messages": [{"role": "user", "content": query2}]},
  config=config
)
print(result2["messages"][-1].content)