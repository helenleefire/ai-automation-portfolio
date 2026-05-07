import asyncio
import itertools
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader 
from pathlib import Path
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma (
  collection_name="script_collection",
  embedding_function= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
  persist_directory="./screenwriting_helper/chroma_langchain_db"
)

folder = Path("./screenwriting_helper/scripts")

async def ingestData():
  patterns = ["*.pdf", "*.txt"]
  for file in itertools.chain(*[folder.glob(p) for p in patterns]):
    if file.suffix == ".pdf" :
      data = UnstructuredPDFLoader(file, strategy="auto").load()
    elif file.suffix == ".txt" :
      data = TextLoader(file).load()
    else:
      continue
    separater_keywords= ["EXT", "ext", "Ext", "INT", "int", "Int"]
    document_splitter = RecursiveCharacterTextSplitter(separators=separater_keywords, chunk_size=10000, chunk_overlap=0)
    documents = document_splitter.split_documents(data)
    await vectorstore.aadd_documents(documents)

asyncio.run(ingestData())