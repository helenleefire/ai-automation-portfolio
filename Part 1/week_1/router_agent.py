import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

# create tool to be used by agent that will classify tickets to different categories
@tool
def classify_ticket(message: str) -> str:
  """ Create a ticket according to the messages submitted and classify the ticket into three categories of
  billing, technical and general. Add a summary of the issue on the ticket.
  Update existing ticket if issues can be combined and looked into at once."""

  return f"Here is the summary of the ticket created: ${message}"

# define/initiate a chat model using claude sonnet
model = init_chat_model("claude-sonnet-4-6")

# define checkpointer to add short term memory to agent
checkpointer = InMemorySaver()


prompt = """
  <role_of_agent>
  You are a customer support routing agent.
  You will classify incoming messages and route them to the correct team
  while responding to customer inquiries in friendly but professional manner.
  </role_of_agent>
  <ticket_format>
    Title: [concise title here]
    Category: [billing | technical | general]
    Summary: [brief summary of the issue]
    Full Conversation: [full conversation thread]
  </ticket_format>
  <instructions>
    Create a separate ticket for each distinct issue rather than combining them
    and keep the title descriptive but concise.
    You can add the conversation to relevant tickets in case human agents need further reference
    but you can expect them to rely mostly on the summary you will output.
  </instructions>
  """

# define agent to call
agent = create_agent(
  model = model,
  tools = [classify_ticket],
  system_prompt= prompt,
  checkpointer=checkpointer
)


if __name__ == "__main__" :
  print ("Is there a problem I can assist with? \n")

  config = {"configurable": {"thread_id": "rag-session-1"}}
  while True :
    question = input ("Your input: ").strip()
    if question.lower() == "done" :
      break
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    print(f"\nAgent: {result['messages'][-1].content}. \nType done to exit.")