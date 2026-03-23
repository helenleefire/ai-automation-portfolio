import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

collection.add(
  ids=["id1", "id2", "id3"],
  documents=[
    "This is a document about python",
    "This is a document about chroma",
    "This is a document about ai"
  ]
)

results = collection.query(
  query_texts=["This a query document about ai"]
)

print(results)