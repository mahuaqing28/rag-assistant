import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import torch

load_dotenv()

# 初始化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载模型时直接指定设备
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("knowledge")

deepseek = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

# 假装这是你的"知识库"
docs = [
    "公司年假政策：入职满一年可享受10天年假，满三年15天。",
    "报销流程：发票需在30天内提交，超期不予报销。",
    "公司地址位于北京市朝阳区某某大厦18层。",
]

# 存进向量库
embeddings = embed_model.encode(docs).tolist()
collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["1", "2", "3"]
)

def rag_query(question: str):
    # 第一步：检索
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    retrieved_docs = results["documents"][0]

    # 第二步：拼Prompt
    context = "\n".join(retrieved_docs)
    prompt = f"""你是一个企业内部助手，只根据以下资料回答问题，不要编造。

资料：
{context}

问题：{question}
"""

    # 第三步：生成
    response = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print(f"\n检索到的文档：{retrieved_docs}\n")
    print("回答：", end="")
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()

# 测试
rag_query("我入职两年了，能休几天年假？")