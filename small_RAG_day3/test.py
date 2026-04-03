# test 相似度
import numpy as np
from sentence_transformers import SentenceTransformer

#embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

q = embed_model.encode(["我入职两年了能休几天年假"])
d1 = embed_model.encode(["报销流程：发票需在30天内提交，超期不予报销。"])
d2 = embed_model.encode(["公司年假政策：入职满一年可享受10天年假，满三年15天。"])

print(cosine_similarity(q[0], d1[0]))
print(cosine_similarity(q[0], d2[0]))