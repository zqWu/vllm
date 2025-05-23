- kv cache 是怎么获取的、如何进行保存、再次读取 
- kv cache 的分配, 物理块 / 逻辑块
	- 分配: worker.cache_engine 初始化时, 有 CacheEngine._allocate_kv_cache(...)-> List[torch.Tensor]
- kv cache 的shape
	- 输入参数
		- num_blocks=计算得到
		- block_size=配置默认16
		- num_kv_heads=模型配置, opt124m有12个注意力头
		- head_size=模型配置, opt124m每个头有64维

	- 不同的各个有所不同
		- paged_attn: shape = return (2, num_blocks, block_size * num_kv_heads * head_size)
	- xFormer: 使用 paged_attn
	- flash_attn: shape = (2, num_blocks, block_size, num_kv_heads, head_size)

- page attn 是如何进行映射管理的
  - vllm plugin system https://docs.vllm.ai/en/latest/design/plugin_system.html
- flash attn 源码, 是怎么进行计算的
- attn-backend是什么? vllm 如何选择 attn-backend, 以及如何进行计算的
  - 模型是如何使用(调用) flash attn 相关函数(或算子)
- vllm中 forward_context.virtual_engine是什么
	- `kv_cache = self.kv_cache[forward_context.virtual_engine]`
	- virtual_engine的存在主要是为了支持pipeline parallelism。
    - 在pipeline parallelism中，不同的pipeline stage可能需要使用不同的虚拟引擎，每个虚拟引擎都有自己独立的KV缓存。
    - 注意力层初始化时会为每个pipeline stage创建独立的KV缓存
- prefill 与 decode 在流程上的细节
- GQA. 在qwen3.config.json中 hidden_size5120 != head_dim128 * num_attention_head64
- step 过程细节
- step中, batch中, 如果一个request结束, 要怎么处理

- tensor parallel是如何做的: 原理上是矩阵乘法的row by row, 代码上 TODO
- tokenizer_group 是什么
- cuda graph 是什么
	- https://zhuanlan.zhihu.com/p/467466998
	实际应用程序中有大量的GPU操作, 每次都由 cpu独立提交到GPU并启动独立计算
		每次提交有启动开销，而且可能还有锁机制
	思想: cuda graphs将整个计算流程定义为一个图, 单次提交, 节省开销
	一种gpu层面的性能优化技术
- mla multi-layer latent attention
- 模型是如何进行加载的: 见流程分析/create_engine.md
- 模型格式 safetensor / pytorch.bin / ...

- 推理日志
```
Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 47.5 tokens/s, Running: 6 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 5.6%, CPU KV cache usage: 0.0%.
```
在 Running: 1 reqs时, 大概是 8token/s
1. 显然这个是并行处理,这种并行处理的细节
2. 对于完成的req, 是等到全部完成后一起返回, 还是完成一个返回一个
