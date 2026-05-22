import asyncio
import itertools
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader 
from pathlib import Path
from agent_tools import reference_store, user_store


reference_scripts = Path("./screenwriting_helper/scripts")

async def ingestData(user_file: bool, file_path: Path) -> bool:
  if user_file:
    if not file_path.exists() :
      print("I wasn't able to find any files")
      return False
    else :
      files = [file_path]
      store = user_store
  else:
    files = itertools.chain(*[reference_scripts.glob(p) for p in ["*.pdf", "*.txt"]])   
    store = reference_store
    
  for file in files:
    if file.suffix == ".pdf" :
      data = UnstructuredPDFLoader(file, strategy="auto").load()
    elif file.suffix == ".txt" :
      data = TextLoader(file).load()
    else:
      continue
    separater_keywords= ["EXT", "ext", "Ext", "INT", "int", "Int"]
    document_splitter = RecursiveCharacterTextSplitter(separators=separater_keywords, chunk_size=10000, chunk_overlap=0)
    documents = document_splitter.split_documents(data)
    await store.aadd_documents(documents)

  return True

# asyncio.run(ingestData())