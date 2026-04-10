import os
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
import asyncio

load_dotenv()

client = AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

async def main() -> None:
    message = await client.messages.create(
        max_tokens=1000,
        messages=[
            {
                "role":"user",
                "content":"how do you learn to drive",
            }
        ],
        model="claude-opus-4-6"
    )
    print(message.content)

asyncio.run(main())