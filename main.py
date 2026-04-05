from pathlib import Path
import os

import chromadb
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "experiment_report.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_knowledge"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# 模型和客户端在模块导入时初始化，避免每次请求重复加载。
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=DEVICE)
reranker = CrossEncoder("BAAI/bge-reranker-base", device=DEVICE)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


class QueryRequest(BaseModel):
    question: str


def load_pdf_text(pdf_path: Path) -> str:
    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise RuntimeError("缺少 PyMuPDF 依赖，无法读取 PDF。请在当前虚拟环境安装 pymupdf。") from exc

    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def ensure_knowledge_base() -> None:
    if collection.count() > 0:
        return
    if not PDF_PATH.exists():
        raise RuntimeError(f"知识库 PDF 不存在：{PDF_PATH}")

    text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))],
    )


def retrieve(question: str) -> list[str]:
    if collection.count() == 0:
        raise HTTPException(status_code=500, detail="知识库为空，无法检索。")

    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)
    docs = results["documents"][0]
    if not docs:
        return []

    pairs = [[question, doc] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:3]]


def stream_answer(question: str, docs: list[str]):
    if not docs:
        yield "知识库中没有检索到相关内容。"
        return

    if not os.getenv("DEEPSEEK_API_KEY"):
        yield "未配置 DEEPSEEK_API_KEY，无法调用大模型生成回答。"
        return

    context = "\n---\n".join(docs)
    prompt = f"""你是一个专业助手，只根据以下资料回答问题，资料中没有的信息就说不知道，不要编造。

资料：
{context}

问题：{question}
"""
    response = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        if content:
            yield content


@app.on_event("startup")
async def startup_event():
    ensure_knowledge_base()


@app.post("/chat")
async def chat(request: QueryRequest):
    docs = retrieve(request.question)
    return StreamingResponse(
        stream_answer(request.question, docs),
        media_type="text/plain",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "chunks": collection.count()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
