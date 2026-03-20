import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

class Summarize(BaseModel) :
  summary: str = Field("Short summary of article")
  key_points: list[str] = Field("Article's key points")
  tone: str = Field("The tone used for summarizing")

model = init_chat_model("claude-sonnet-4-6")

agent = create_agent(
  model=model,
  system_prompt="Return a structured and concise summary",
  response_format= Summarize
)