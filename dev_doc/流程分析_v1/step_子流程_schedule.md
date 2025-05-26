# 文件 vllm/v1/core/sched/scheduler.py, 类 Scheduler::schedule

# 概念
连续批处理：支持动态批处理，可以在不同处理阶段混合请求
前缀缓存：利用已计算的 token 减少重复计算
多模态支持：处理编码器输入和预算管理
推测解码：支持投机 token 的调度 speculative decoding
抢占机制：在内存不足时智能抢占低优先级请求

# 概念
request.num_computed_tokens
request.num_tokens = request.num_prompt_tokens
spec_token = speculative_token, 目前一律按照=0来理解

num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).

scheduler.max_num_scheduled_tokens
scheduler.max_num_encoder_input_tokens

scheduler.num_lookahead_tokens
scheduler.num_spec_tokens

num_scheduled_tokens

