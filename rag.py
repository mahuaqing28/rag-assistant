import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import torch
from sentence_transformers import CrossEncoder
load_dotenv()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)


embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("pdf_knowledge")

deepseek = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

def load_and_chunk(pdf_path: str, chunk_size=200, overlap=50):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    chunks = []
    start = 0
    while start < len(full_text):
        chunks.append(full_text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

def build_index(pdf_path: str):
    chunks = load_and_chunk(pdf_path)
    embeddings = embed_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )
    print(f"索引建立完成，共{len(chunks)}块")

def rag_query(question: str):
    # 第一步：Embedding粗召回
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    retrieved_docs = results["documents"][0]

    # 第二步：Reranker精排
    pairs = [[question, doc] for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, retrieved_docs), reverse=True)
    top_docs = [doc for _, doc in ranked[:3]]

    # 第三步：拼Prompt生成
    context = "\n---\n".join(top_docs)
    prompt = f"""你是一个专业助手，只根据以下资料回答问题，资料中没有的信息就说不知道，不要编造。

资料：
{context}

问题：{question}
"""

    response = deepseek.chat.completions.create(
        model="kimi-k2.5",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print(f"\nReranker排序后Top3文档：")
    for i, (score, doc) in enumerate(ranked[:3]):
        print(f"[{i+1}] score={score:.4f} | {doc[:50]}...")

    print("\n回答：", end="")
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()

""" def rag_query(question: str):
    # 检索
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    retrieved_docs = results["documents"][0]

    # 拼Prompt
    context = "\n---\n".join(retrieved_docs)
    prompt = f""""""你是一个专业助手，只根据以下资料回答问题，资料中没有的信息就说不知道，不要编造。

资料：
{context}

问题：{question}

"""
"""
    # 生成
    response = deepseek.chat.completions.create(
        model="kimi-k2.5", # deepseek-chat
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print(f"\n检索到的文档片段：")
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i+1}] {doc[:50]}...")
    
    print("\n回答：", end="")
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print() """

# 运行
build_index("experiment_report.pdf")

while True:
    question = input("\n请输入问题（输入q退出）：")
    if question == "q":
        break
    rag_query(question)



