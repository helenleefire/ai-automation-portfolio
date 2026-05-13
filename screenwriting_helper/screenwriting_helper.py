from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from ingest_data import vectorstore, embedding
from setting import setting
from generate_scene_outline import generate_scene_outline 
@tool 
def data_retriever (query: str) -> str :
    """Use this tool to reference scripts of award movies to ensure the advice
    given will be of quality and end product will be well received by critics 
    and the general audience"""
    try:
        query_vector = embedding.embed_query(query)
        search = vectorstore.max_marginal_relevance_search_by_vector(
            k=5, fetch_k=20, embedding=query_vector)
        if not search:
            return "I don't think I can help you using my knowledge base"
        return "\n".join([doc.page_content for doc in search])
    except:
        return "I'm unable to refer to my script knowledgebase"

agent = create_agent(
    model=init_chat_model(model=setting.model, api_key=setting.anthropic_api),
    tools=[data_retriever, generate_scene_outline],
    checkpointer= InMemorySaver(),
    system_prompt="""You are a screenwriting helper agent. You help screenwriters polish
    their ideas and scripts as well as give suggestions for how to continue story, analyze
    given scripts and give advice on how to keep consistency and pace stories. You will refer
    back to award winning scripts in your database to give advice on how to write a bette script.
    You will also help generate scenes for unfinished scripts. Try to get the user to give you
    as much detail as possible and try to be specific about how the user should improve their script."""
)

if __name__ == "__main__":
    print("I am a screenwriting assistant that can help you with generating a scene for " \
    "your script. Give me a brief description of your story and " \
    "what you are trying to achieve. We'll take it off from there! \nType stop to exit.")
    while True:
        query = input("Your input: ").strip()
        if query.lower() == "stop":
            break
        result = agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            config={"configurable":{"thread_id":"rag-session-1"}}
        )
        print(f"\nAgent:{result['messages'][-1].content} \nType stop to exit.")i