import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import chromadb

def load_pdf(pdf_path: str):
    # 提取文字
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap让相邻块有重叠
    return chunks

# 测试
text = load_pdf("experiment_report.pdf")
chunks = chunk_text(text)
print(f"总共切了{len(chunks)}块")
print(f"第一块内容：{chunks[0]}")