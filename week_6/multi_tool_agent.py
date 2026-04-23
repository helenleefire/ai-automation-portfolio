# Define lookup_customer (mock DB from Week 3), 
# search_knowledge_base (your Week 2 RAG retriever), 
# and escalate_ticket (logs to file). Confirm each works independently. 
# Write strong docstrings — the agent reads these to decide which tool to call.
from config import setting
from anthropic import Anthropic
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


client = Anthropic(api_key=setting.anthropic_api)

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

class SupportTicket(BaseModel) :
  title: str = Field("Concise title of the issue")
  category: str = Field("One of: billing, technical, general")
  summary: str = Field("The summary of issue in one to three sentences")
  chat_history: list[ChatHistory] 
  priority: str = Field("One of: low, medium, high, urgent")

vectorstore = Chroma (
  collection_name="my_collection",
  embedding_function= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
  persist_directory="./chroma_langchain_db"
)

@tool
def data_retriever(query: str) -> str:
    """Retrieve relevant information from existing documents..."""                                                     
    try:                                                          
        search = vectorstore.similarity_search(query)                                                                  
        if not search:                               
            return "I don't see relevant information in my knowledge base"                                             
        return "\n".join([doc.page_content for doc in search])            
    except:                                                                                                            
        return "I'm sorry, I wasn't able to access my knowledge base"                                                  
@tool
def lookup_user(user_id: str) -> str:
  """Look up user information using their user ID if you think your answer
  would benefit from knowing the background of the user.
  Give appropriate response according to user information.
  If user information cannot be found or information is insufficient, 
  be conservative with your response and don't overpromise"""
  user = USER_DB.get(user_id)
  if not user :
    return f"User id of ${user_id} not found in database"
  return (
    f"Name: {user['name']} "
    f"Employee Status: {user['employee_status']}"
    f"Department: {user['department']}"
    f"Flag: {user['flag']}"
  )

@tool
def classify_ticket(classification: str) -> str:
  """ Create a ticket according to the messages submitted and classify the 
  ticket into three categories of billing, technical and general. 
  Add a summary of the issue on the ticket. Update existing ticket 
  if issues can be combined and looked into at once."""

  return f"Ticket classification: ${classification}"

@tool
def escalate_ticket(reason: str) -> str:
  """Escalate a support ticket to a human agent if the matter 
  needs to be dealt with caution or if the customer is frustrated 
  with the support they are getting"""
  import logging
  logging.warning(f"Escalation: {reason}")
  return f"Ticket esclated for reason: {reason}"

@tool
def create_ticket(title: str, category: str, summary: str, priority: str) -> str:
    """Create a ticket to track customer inquiries..."""                                                               
    return f"Ticket created - Title: {title}, Category: {category}, Priority: {priority}, Summary: {summary}"

agent = create_agent(
  model=init_chat_model(model=setting.model, api_key=setting.anthropic_api),
  tools=[data_retriever, lookup_user, classify_ticket, create_ticket, escalate_ticket],
  checkpointer=InMemorySaver(),
  system_prompt="""Use the tool in appropriate settings. 
  Don't shy away from using the tools as using them will help you provide support in
  leadership approved ways. Be friendly to the customer but also be conservative
  in the answers you give. When you create tickets, let the user know of it and its content."""
)

if __name__ == "__main__" :  
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
        config={"configurable": {"thread_id": "rag-session-1"}}
    )

    conversation = "\n".join(                                                                                             
        f"{m.type}: {m.content}"                                                                                          
        for m in result['messages']                                                                                       
    )               


    print(f"\nAgent: {result['messages'][-1].content}. \nType done to exit.")