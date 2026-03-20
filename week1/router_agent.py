import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

@tool
def classify_ticket(message: str) -> str:
  """ Create a ticket according to the messages submitted and classify the ticket into three categories of
  billing, technical and general. Add a summary of the issue on the ticket."""

  return f"Classifying ${message}"


model = init_chat_model("claude-sonnet-4-6")
checkpointer = InMemorySaver()

agent = create_agent(
  model = model,
  tools = [classify_ticket],
  system_prompt= """Respond appropriately to the concerns""",
  checkpointer=checkpointer
)
