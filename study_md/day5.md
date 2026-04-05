# DAY5 FAST API 和 CrossEncoder Reranker

## CrossEncoder Reranker 和embedding

Embedding（双塔/双编码器）：

Query 和 Chunk 分别输入到两个独立（但通常权重共享）的编码器。

输出各自的稠密向量。

通过余弦相似度或点积等向量距离计算相关性。

优点：可以预先计算所有 Chunk 的向量，查询时速度极快。

缺点：Query 和 Chunk 之间缺乏深层交互，可能丢失细粒度匹配信号（如否定词、短语顺序）。

CrossEncoder（交叉编码器，常用作 Reranker）：

Query 和 Chunk 拼接成一个序列，一起输入同一个编码器（通常是 BERT 类模型）。

模型输出一个相关性分数（比如 0~1 或 logits）。

优点：能进行充分的 token 级交互，更准确地捕捉语义匹配、逻辑关系、细微差别。

缺点：无法预计算（因为必须实时拼接 Query 与每个 Chunk），计算量大、速度慢，不适合直接用于召回。

## FAST API

使用 StreamingResponse 转发获得的流式输出