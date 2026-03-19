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

ticket = {
  "content" : "My invoice shows the wrong amount"
}

result = agent.invoke(
  {"messages": [{"role":"user",
                 "content":f"{ticket["content"]}"}]},
  {"configurable":{"thread_id":"1"}}
)

print(result["messages"][-1].content)

ticket2 = {
  "content" : "Can you tell me more about what I just asked?"
}

result2 = agent.invoke(
  {"messages": [{"role":"user",
                 "content":f"{ticket2["content"]}"}]},
  {"configurable":{"thread_id":"1"}}
)

print(result2["messages"][-1].content)


ticket3 = {
  "content" : "How long should I wait for a response??"
}

result3 = agent.invoke(
  {"messages": [{"role":"user",
                 "content":f"{ticket3["content"]}"}]},
  {"configurable":{"thread_id":"1"}}
)

print(result3["messages"][-1].content)
