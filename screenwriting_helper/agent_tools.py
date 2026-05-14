import asyncio
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

reference_store = Chroma (
  collection_name="script_collection",
  embedding_function= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
  persist_directory="./screenwriting_helper/chroma_langchain_db"
)

user_store = Chroma (
  collection_name="user_scripts",
  embedding_function= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
  persist_directory="./screenwriting_helper/chroma_langchain_db"
)

@tool 
def data_retriever (query: str) -> str :
    """Use this tool to reference scripts of award movies to ensure the advice
    given will be of quality and end product will be well received by critics 
    and the general audience"""
    try:
        query_vector = embedding.embed_query(query)
        search = reference_store.max_marginal_relevance_search_by_vector(
            k=5, fetch_k=20, embedding=query_vector)
        if not search:
            return "I don't think I can help you using my knowledge base"
        return "\n".join([doc.page_content for doc in search])
    except:
        return "I'm unable to refer to my script knowledgebase"
    
@tool
def analyze_structure(analysis: str) -> str:
    """Read the script given by the user and analyze its structure againt 
    the three act framework of well regarded movie scripts in the
    knowledge base. Summarize what happens in each act of the given script
    and make a judgement on how they fare against the three act structure
    of good movies. Give cues on which part of the story should be elaborated
    and which part can be shortened for better momentum. Evaluate the 
    rhythm of how the story progress and give advice on how to improve the 
    structure and and pacing."""
    return analysis

@tool
def analyze_characters (analysis: str) -> str:
    """Find the main characters and the main supporting characters of the
    script and retrieve the scenes that each of these characters are in.
    Figure out the motivations and the characterisitcs of these characters
    and give brief descriptions about them. Judge if the ways these characters 
    act and speak are consistent. If not, give examples of scenes the characters 
    act out of their own character and give advice on how to keep consistnency, 
    or better convey their motivations. You can also provide examples on how
    other movies convey a character's motivations or showcase their characters."""
    return analysis

@tool
def generate_scene_outline (outline: str) -> str :
    """Create an outline for a scene that you think will fit 
    in to the script the user is trying to write. Reference past conversations with the user to
    make sure the scene outline generated would make sense in the context of everything
    about this story and its characters. Try to get as much detail from the writer and explain
    the choices you made in creating your outline and why you think it would work for
    the story they're trying to write."""
    return outline