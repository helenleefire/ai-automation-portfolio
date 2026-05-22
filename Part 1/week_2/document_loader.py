from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import asyncio

url = [
  "https://docs.oracle.com/javase/8/docs/api/java/util/HashMap.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html",
  "https://docs.oracle.com/javase/8/docs/api/java/util/Hashtable.html"
  ]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma (
  collection_name="my_collection",
  embedding_function=embeddings,
  persist_directory="./chroma_langchain_db"
)

async def ingestData():
  web_data = WebBaseLoader(url).load()
  document_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  data = document_splitter.split_documents(web_data)
  await vectorstore.aadd_documents(data)

asyncio.run(ingestData())