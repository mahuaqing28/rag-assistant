# DAY 2 学习笔记
运行 rag.py
```python
embeddings = model.encode(sentences)

print(f"向量维度：{embeddings.shape}")
print(f"第一个句子的向量（前10个数字）：{embeddings[0][:10]}")
```
输出结果：
```plaintext
向量维度：(3, 384)
第一个句子的向量（前10个数字）：[-0.42558548  0.67283463  0.2002414  -0.21243054 -0.8126502  -0.30423877
  1.0851915   0.03313304  0.9195695  -0.07800552]
```

- 延伸1：sentence-transformers基于PyTorch，模型本体是个小型BERT。PyTorch底层的矩阵运算是C++/CUDA写的。

- 延伸2：chromaDB 使用什么样的算法建立索引来组织？ 是HNSW算法，HNSW是一种近似最近邻算法，牺牲一点点精度，换取在百万级向量里毫秒级检索的速度。

- 延伸3：为什么要"tolist()" ,encode后不是list吗？ 因为类型不同一个是nummpy库中的数据类型，一个是python原生的list

- 延伸4： n_results=2 代表了什么？ 返回的对象（向量）的个数

- 断点： 可以换成GPU执行吗？