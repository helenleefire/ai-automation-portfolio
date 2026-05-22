import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.factory import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from document_loader import vectorstore

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["USER_AGENT"] = "my-app/1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@tool
async def data_retriever(query: str) -> str:
  """Reterieve relevent informtaion from existing documents to answer questions.
  Let the user know when using them."""
  try :
    search = await vectorstore.asimilarity_search(query)
    if not search :
      return "I don't see relevant information in my knowledge base"
    return "\n".join([doc.page_content for doc in search])
  except :
    return "I'm sorry, I wasn't able to access my knowledge base"

model = init_chat_model("claude-sonnet-4-6")

checkpointer = InMemorySaver()

agent = create_agent(
  model=model,
  tools=[data_retriever],
  checkpointer=checkpointer,
  system_prompt="""Use the data retrieval tool when answering questions.
  Keep the response to at most a paragraph unless user asks for more detail"""
)