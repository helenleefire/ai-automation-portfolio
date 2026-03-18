import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

@tool
def classify_ticket(ticketTitle: str, ticketContent: str) -> str:
  """ Classify the ticket into three categories of
  billing, technical and general

  Args:
    ticketTitle: title of ticket
    ticketContent: content of ticket
  """

  return f"Ticket {ticketTitle} with description {ticketContent} was categorized"


model = init_chat_model("claude-sonnet-4-6")

agent = create_agent(
  model = model,
  tools = [classify_ticket],
  system_prompt= """Be concise and try to respond in one sentence"""
)

ticket = {
  "title" : "My computer broke",
  "content" : "My computer stopped turning on and it's plugged in"
}

result = agent.invoke(
  {"messages": [{"role":"user",
                 "content":f"What is the category of ticket {ticket["title"]} with description {ticket["content"]}"}]})

print(result["messages"][-1].content)