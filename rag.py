from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

sentences = [
    "今天天气很好",
    "阳光明媚，适合出门",
    "我喜欢吃火锅",
]

embeddings = model.encode("今天天气很好")

print(f"向量维度：{embeddings.shape}")
# print(f"第一个句子的向量（前10个数字）：{embeddings[0][:10]}")
print(f"第一个句子的向量（前10个数字）：{embeddings[:10]}")
# extend
import numpy as np

def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

sim_12 = cosine_similarity(embeddings[0], embeddings[1])
sim_13 = cosine_similarity(embeddings[0], embeddings[2])

print(f"天气 vs 阳光：{sim_12:.4f}")
print(f"天气 vs 火锅：{sim_13:.4f}")