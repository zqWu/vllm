# 有关 kv cache
- kv cache 是怎么获取的、如何进行保存、再次读取 
- kv cache 的分配, 物理块 / 逻辑块
	- 分配: worker.cache_engine 初始化时, 有 CacheEngine._allocate_kv_cache(...)-> List[torch.Tensor]
- kv cache 的shape
	- 输入参数
		- num_blocks=计算得到
		- block_size=配置默认16
		- num_kv_heads=模型配置, opt124m有12个注意力头
		- head_size=模型配置, opt124m每个头有64维

# 是什么 
attention 计算过程中
model 权重包含 Wq, Wk, Wv, 一般是shape=(dim, dim)

当有一个 token (tN) 进入 attention 层时
总的token是这样序列 [t1, ... tN-1, tN] = (N,1) shape
经过 embedding + pe 之后, 得到 tensor (N, dim)
每个 token_k 对应 tensor[k,:]

embed * Wq 得到的就是 q
embed * Wk 得到的就是 k
embed * Wv 得到的就是 v

# 在 vllm 中, kv cache是怎么 读写的

kv cache是一个分布在 GPU上的 tensor
初始化时 kv_cache = torch.zeros(kv_cache_allocation_shape, device=GPU).permute(kv_cache_stride_order)
kv_cache_allocation_shape, kv_cache_stride_order
这2个值是 attn_backend有关的参数, 设置 kv_cache存储的形状, 以及处理方式

## 读写过程
完全由 attention 来决定
在 Attention类初始化时, 就会拿到 kv_cache


```python
# vllm.attention.layer.Attention
class Attention(nn.Module):
	...
	def __init__(self, ...):
		self.kv_cache_dtype = kv_cache_dtype
		self._k_scale_float = 1.0
        self._v_scale_float = 1.0
		# 每个 pp 分配不同的 kv_cache
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config(
            ).parallel_config.pipeline_parallel_size)
        ]

	def forward(self, ):
		self_kv_cache = self.kv_cache[forward_context.virtual_engine]
		self.impl.forward(self, ..., self_kv_cache)
		# 在真正做 forward attn_backend相关的类里面 如 FlashAttentionImpl

class FlashAttentionImpl(AttentionImpl):
	...
	def forward(self, ..., kv_cache: torch.Tensor,):
		# 将新的 key, value 存储到 kv-cache中
		key_cache = kv_cache[0]
		value_cache = kv_cache[1]
		torch.ops._C_cache_ops.reshape_and_cache_flash( 
			key,
			value,
			kv_cache[0],
			kv_cache[1],
			updated_slot_mapping.flatten(),  # type: ignore[union-attr]
			kv_cache_dtype,
			layer._k_scale,
			layer._v_scale,
		)
#
```

# k_cache的使用
在 flash_attn.py 的 FlashAttentionBackendImpl中，
flash_attn_varlen_func方法参数传递了 k_cache, 和 block_table, 

为什么不需要 slot_mapping这个参数？
block_table中并不是所有的 slot都是有效k_cache, 有些位置还空着未占用

## 回答
见 kv_cache_使用2.png
seqused_k 来源是调度时进行的计算，放到 attn_metadata中了
```python
flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache, <===================== 整块内存
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k, <============= seq长度, 如  [19, 8]
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table, <========= 整块内存中使用了哪些 block, 如  [[2,3,0], [4,0,0]]
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=layer._q_scale.expand(descale_shape),
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
```