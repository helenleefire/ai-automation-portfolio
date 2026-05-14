import asyncio
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from setting import setting
from agent_tools import data_retriever, generate_scene_outline, analyze_characters, analyze_structure
from ingest_data import ingestData
from pathlib import Path

agent = create_agent(
    model=init_chat_model(model=setting.model, api_key=setting.anthropic_api),
    tools=[data_retriever, generate_scene_outline, analyze_characters, analyze_structure],
    checkpointer= InMemorySaver(),
    system_prompt="""You are a screenwriting helper agent. You help screenwriters polish
    their ideas and scripts as well as give suggestions for how to continue story, analyze
    given scripts and give advice on how to keep consistency and pace stories. You will refer
    back to award winning scripts in your database to give advice on how to write a bette script.
    You will also help generate scenes for unfinished scripts. Try to get the user to give you
    as much detail as possible and try to be specific about how the user should improve their script.
    If the script is not provided by the user, you are not expected to use analyze_characters, analyze_structure
    tools."""
)

if __name__ == "__main__":
    file_path = input("""
        Hello! I am a screenwriting assistant! \n
          If you have a script you would like me to review, please provide its file path.\n
          Type none if you don't have any scripts to provide. \n
          Your input: 
    """)
    if file_path.strip() != "none":
        asyncio.run(ingestData(True, Path(file_path)))

    print("Tell me what you need help with today and we can start from there!")
    while True:
        query = input("Your input: ").strip()
        if query.lower() == "stop":
            break
        result = agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            config={"configurable":{"thread_id":"rag-session-1"}}
        )
        print(f"\nAgent:{result['messages'][-1].content} \nType stop to exit.")