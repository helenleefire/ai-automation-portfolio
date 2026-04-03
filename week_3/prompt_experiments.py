import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

# Mock up user database
USER_DB = {
  "1" :{"name": "Sarah Johnson", "employee_status": "Full Time", "department": "HR", "flag": "Suspicious"},
  "2" :{"name": "Joseph Adams", "employee_status": "Part Time", "department": "IT", "flag": None},
  "3" :{"name": "Alex Jones", "employee_status": "Retiree", "department": "Finance", "flag": "Prioritize"},
  "4" :{"name": "Samantha Button", "employee_status": "Past Employee", "department": "Engineering", "flag": None},
}

class ChatHistory(BaseModel) :
  user_input: str = Field("Input from user")
  agent_response: str = Field("Full response given to user")

# class for structured output for tickets
class SupportTicket(BaseModel) :
  title: str = Field("Concise title of the issue")
  category: str = Field("One of: billing, technical, general")
  summary: str = Field("The summary of issue in one to three sentences")
  chat_history: list[ChatHistory] 
  priority: str = Field("One of: low, medium, high, urgent")
  
# tool to be used by agent that will classify tickets to different categories
@tool
def classify_ticket(message: str) -> str:
  """ Create a ticket according to the messages submitted and classify the ticket into three categories of
  billing, technical and general. Add a summary of the issue on the ticket.
  Update existing ticket if issues can be combined and looked into at once."""

  return f"Here is the summary of the ticket created: ${message}"

# tool help create more tailored responses and tickets and allow better prioritzation
@tool
def lookup_user(user_id: str) -> str:
  """Look up user information using their user ID. Give appropriate response according to user information.
  If user information cannot be found or information is insufficient, be conservative with your response and don't overpromise"""
  user = USER_DB.get(user_id)
  if not user :
    return f"User id of ${user_id} not found in database"
  return (
    f"Name: {user['name']} "
    f"Employee Status: {user['employee_status']}"
    f"Department: {user['department']}"
    f"Flag: {user['flag']}"
  )
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
  <chain_of_thought>
    Think step by step before classifying.
  </chain_of_thought>
  <examples>
    <scenario_1>
      <user_input>
        my most recent paystub had amount that is lower than last cycle's
      </user_input>
      <output>
        **Issue:** Your most recent paystub shows a lower amount than the previous pay cycle.
        **Team:** Billing
      </output>
    </scenario_1>
    <scenario_2>
      <user_input>
        i also didn't receive my bonus and i would like to receive the back pay together
      </user_input>
      <output>
        **Issue:** You did not receive your expected bonus payment. You've also requested that this be resolved alongside your lower-than-usual paystub issue, so that both can be paid out together as back pay.
        **Team:** Billing
        **Note** A new ticket has been created but issues will be linked so both can be received together as back pay.
      </output>
    </scenario_2>
  </examples> 
  <ticket_format>
    Title: [concise title here]
    Category: [billing | technical | general]
    Summary: [brief summary of the issue]
    Full Conversation: [full conversation thread]
  </ticket_format>
  <instructions>
    If the user has a "Suspicious" flag, be conservative and do not make promises.
    If the user has a "Prioritize" flag, escalate their ticket priority accordingly.                    
    If the user is a "Past Employee", remind them that support may be limited.      
    You can combine tickets if they are related but create a separate ticket for each distinct issues.
    Keep the title descriptive but concise.
    You can add the conversation to relevant tickets in case human agents need further reference
    but you can expect them to rely mostly on the summary you will output.
  </instructions>
  """

# define agent to call
agent = create_agent(
  model = model,
  tools = [classify_ticket, lookup_user],
  system_prompt= prompt,
  checkpointer=checkpointer
)


if __name__ == "__main__" :  
  config = {"configurable": {"thread_id": "rag-session-1"}}
  employeeId = input("Please provide your employee ID so I can provide better help.\n").strip()
  print("What can I help you with?\n")
  first_message = True
  while True :
    question = input ("Your input: ").strip()
    if question.lower() == "done" :
      break

    content = f"[User ID: {employeeId}] {question}" if first_message else question
    first_message = False
    result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config=config,
    )

    conversation = "\n".join(                                                                                             
        f"{m.type}: {m.content}"                                                                                          
        for m in result['messages']                                                                                       
    )               
    ticket = model.with_structured_output(SupportTicket).invoke(conversation)

    print(f"\nAgent: {result['messages'][-1].content}. \nType done to exit.")
    print(f"\nUpdate: {ticket}")