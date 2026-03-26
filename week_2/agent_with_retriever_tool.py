import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

"""
To do

This is the core build of the week.
Wrap your Chroma retriever as a @tool, pass it to create_agent.
Ask 5 questions over your document.
Observe when the agent uses the tool vs answers from training.
The tool docstring is critical —
the agent reads it to decide when to retrieve.

What to test
Ask 5 questions over your document — mix these types:

A question clearly answered in the document
A question partially answered in the document
A question not in the document at all
An ambiguous question that could go either way
A follow-up to one of the above

"""
load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

model = init_chat_model("claude-sonnet-4-6")

url = ["https://www.google.com"]
web_data = WebBaseLoader(url).load()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
document_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
data = document_splitter.split_documents(web_data)

vectorstore = Chroma (
  collection_name="my_collection",
  embedding_function=embeddings,
  persist_directory="./chroma_langchain_db"
)

vectorstore.add_documents(data)

@tool
def data_retriever(query: str) -> str:
  """Reterieves relevent informtaion from the web document to answer questions"""
  results = vectorstore.similarity_search(query)
  if not results :
    return "I don't see relevant information in my knowledge base"
  return "\n".join([doc.page_content for doc in results])


agent = create_agent(
  model=model,
  tools=[data_retriever],
  system_prompt="""Use the data retrieval tool when answering questions"""
)

result = agent.invoke(
  {"messages": [{"role": "user", "content": "why is the color orange called orange"}]}
)

print(result)
