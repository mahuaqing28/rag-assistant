import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

print(model.device)

client = chromadb.Client()
collection = client.create_collection("test")

docs = [
    "今天天气很好",
    "阳光明媚，适合出门",
    "我喜欢吃火锅",
]

embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["1", "2", "3"]
)

# 用一个新句子去查
query = "今天适合外出吗"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print(results["documents"])

print(results)