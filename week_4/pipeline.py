import os
import anthropic
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from anthropic import AsyncAnthropic
import config

load_dotenv()
config

client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
logging.basicConfig(filename='pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class APIResponse(BaseModel) :
    title : str = Field(description="Title of response in a short sentence")
    summary : str = Field(description="Summary of the answer given to user in at most 3 sentences but shorter the better")
    answer : str = Field(description="Full answer for the user that is at most two paragraphs but shorter the better")

async def main() -> None:
    print ("Ask anything to your agent. Enter stop to end. \n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "stop" :
            break
        try:
            logger.info(f'Calling API, question asked: {question}')
            answer = await client.messages.create(
                max_tokens=1000,
                messages=[
                    {
                        "role":"user",
                        "content": question,
                    }
                ],
                model="claude-opus-4-6",
                tools=[{
                    "name": "format_response",
                    "description": "Format the response",
                    "input_schema": APIResponse.model_json_schema()
                }],
                tool_choice={"type": "tool", "name": "format_response"}
            )
            logger.info(f'Success, input token used: {answer.usage.input_tokens} and output token used: {answer.usage.output_tokens}')
            answerJSON = APIResponse(**answer.content[0].input)
            print(f"\nAgent answer: Title: {answerJSON.title}\nSummary: {answerJSON.summary}\nFull Response: {answerJSON.answer}")
        except (anthropic.AuthenticationError, TypeError) as e :
            logger.error(f'Authentication failed\n {e}')
        except (anthropic.APIConnectionError, TypeError) as e:
            logger.error(f'Server could not be reached\n {e}')
        except (anthropic.RateLimitError, TypeError) as e :
            logger.error(f'Rate limit error was encountered\n {e}')
        print("Enter stop to end")
        
asyncio.run(main())