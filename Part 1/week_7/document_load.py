from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma (
  collection_name="my_collection",
  embedding_function= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
  persist_directory="./chroma_langchain_db"
)

def ingestData():
   data = TextLoader(file_path="./week_6/sample_data/corporate_guideline.txt").load_and_split()
   vectorstore.add_documents(data)
   print("success!")

ingestData()