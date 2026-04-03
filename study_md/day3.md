# DAY3 RAG最小系统的数据流动

## 实验过程
RAG 的组成是：向量数据库，sentense embedding model,LLM. 整个过程的数据流是：

```plaintext
用户提问
   ↓
问题被encode成向量
   ↓
向量库找最相似的文档片段
   ↓
文档片段+问题拼成Prompt
   ↓
DeepSeek生成回答
   ↓
流式返回给用户
```
## question
当使用模型为：paraphrase-MiniLM-L6-v2 时，其认为与查询年假的请求最相关的句子反而是报销流程，这说明该模型在理解中文语义时，存在问题；换用：BAAI/bge-small-zh-v1.5 时，向量空间中，距离请求最接近的句子变成了年假相关的政策，这说明不同模型在中文语义识别上有重大的差异。
