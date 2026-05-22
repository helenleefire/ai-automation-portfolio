import asyncio
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from setting import setting
from agent_tools import data_retriever, user_script_retreiver, generate_scene_outline, analyze_characters, analyze_structure
from ingest_data import ingestData
from pathlib import Path

agent = create_agent(
    model=init_chat_model(model=setting.model, api_key=setting.anthropic_api),
    tools=[data_retriever, generate_scene_outline, analyze_characters, analyze_structure, user_script_retreiver],
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
    valid_file_path = False
    script_provided = False
    script_name = ""
    while not valid_file_path:
        file_path = input("""\nHello! I am a screenwriting assistant!\n
If you have a script you would like me to review, please provide its file path.\n
Type none if you don't have any scripts to provide. \n
Your input: """)
        if file_path.strip() != "none":
            if asyncio.run(ingestData(True, Path(file_path))) == True:
                valid_file_path = True
                script_provided = True
                script_name = Path(file_path).name
        else:
            valid_file_path = True

    if script_provided:
        result = agent.invoke(
            {"messages": [{"role": "user", 
                           "content": f"I have uploaded my script '{script_name}' \
                            for you to review. Use user_script_retreiver to access it\
                                  whenever you need to reference or analyze it."}]},
            config={"configurable": {"thread_id": "rag-session-1"}}
        )
        print(f"\nAgent: {result['messages'][-1].content}\nType stop to exit.")
    else:
        print("Tell me what you need help with today and we can start from there! Type stop to exit.")

    while True:
        query = input("Your input: ").strip()
        if query.lower() == "stop":
            break
        result = agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            config={"configurable":{"thread_id":"rag-session-1"}}
        )
        print(f"\nAgent:{result['messages'][-1].content} \nType stop to exit.")