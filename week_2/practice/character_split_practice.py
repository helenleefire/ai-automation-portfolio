from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



loader = Docx2txtLoader("week_2/practice/sample_data/ipsum.docx")
data = loader.load()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)

vectorstore = Chroma(
  collection_name="my_collection",
  embedding_function=embeddings,
  persist_directory="./chroma_langchain_db",
)

vectorstore.add_documents(texts)

print(texts[0])