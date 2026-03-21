import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

# schema defining the structurd output format, BaseModel - returns validated Pydantic instance
class Summarize(BaseModel) :
  summary: str = Field("Short summary of article")
  key_points: list[str] = Field("Article's key points")
  tone: str = Field("The tone used for summarizing")

model = init_chat_model("claude-sonnet-4-6")

agent = create_agent(
  model=model,
  system_prompt="Return a structured and concise summary",
  # control how agent returns structured data, toolstrategy using tool calling
  response_format= ToolStrategy(Summarize)
)

result = agent.invoke({
  "messages" : [{"role": "user", "content":"Summarize this article: https://docs.langchain.com/oss/python/langchain/philosophy"}]
})

print(result["messages"][-1].content)