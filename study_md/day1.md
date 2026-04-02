# DAY 1 API调用和字段解析
## response 字段解析
> ChatCompletion(id='c8c57452-c167-4f07-83f5-5edd70f754b1', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='Hello! How can I assist you today?', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], created=1775010122, model='deepseek-chat', object='chat.completion', service_tier=None, system_fingerprint='fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache', usage=CompletionUsage(completion_tokens=9, prompt_tokens=10, total_tokens=19, completion_tokens_details=None, prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0), prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=10))

- 该对象包含的字段有：id,choices,created,model,object,service_tier,system_fingerprint,usage,prompt_tokens,total_tokens,completion_tokens_details,prompt_tokens_details,prompt_cache_hit_tokens,prompt_cache_miss_tokens
   - 其中：Choice 包含：finish_reason,index,logprobs,message
     - ChatCompletionMessage 包含：content,refusal, role,annotations,audio,function_call, tool_calls

## RAG 为什么多用流式？
  TTFT（首字响应时间）不一样，RAG的流程是：用户问 → 检索文档 → 拼Prompt → 模型生成 → 返回答案。
  由于总时间长，为了提升用户体验，使用流式响应