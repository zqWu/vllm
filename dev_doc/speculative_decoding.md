https://docs.vllm.ai/en/stable/features/spec_decode.html

应该是一种 推理加速的技术, 暂且知道有这个东西
原理不懂, 以下是 介绍

In speculative decoding, 
a smaller "draft" model helps a larger target llm generate text faster by making initial predictions. 
The draft model generates a batch of tokens, 
and the target LLM then either accepts or refines these predictions, 
allowing the LLM to process multiple tokens concurrently. 

This approach reduces the computational load on the target LLM, speeding up inference. 
