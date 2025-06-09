# 是什么

1. attn backend 是 模型层, 关于 attention 的一种实现

``` python
class AttentionBackend(ABC):
	FlashAttentionBackend
	XFormersBackend
	TritonAttentionBackend
	FlashInferBackend
	... 其他还有10+种实现
	get_kv_cache_shape(): kv_cache的形状

@cache
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    is_blocksparse: bool = False,
    use_v1: bool = False,
    use_mla: bool = False,
) -> Type[AttentionBackend]:
	...

def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    is_blocksparse: bool = False,
    use_mla: bool = False,
) -> Type[AttentionBackend]:
	...


class Attention(nn.Module):
	def __init__(...):
        attn_backend = get_attn_backend(head_size,
                                dtype,
                                kv_cache_dtype,
                                block_size,
                                is_attention_free,
                                blocksparse_params is not None,
                                use_mla=use_mla)


class Qwen3Attention(nn.Module):
    def __init__(...)
        self.attn = Attention(self.num_heads,
                          self.head_dim,
                          self.scaling,
                          num_kv_heads=self.num_kv_heads,
                          cache_config=cache_config,
                          quant_config=quant_config,
                          prefix=f"{prefix}.attn",
                          attn_type=attn_type)

```

# attn-backend 选择过程
```python
def _cached_get_attn_backend(
    head_size: int,                 <====== 从 model.config 得到
    dtype: torch.dtype,             <====== 从 model.config 得到
    kv_cache_dtype: Optional[str],  <====== 从 model.config 得到
    block_size: int,                <====== vllm配置 默认16
    is_attention_free: bool,        <====== 无 attn? 一般 False
    is_blocksparse: bool = False,   <====== block-sparse attention, 一种滑动窗口注意力机制
    use_v1: bool = False,           <====== 默认 true VLLM_USE_V1
    use_mla: bool = False,          <====== deepseek的 multi-head latent attn
) -> Type[AttentionBackend]:

1. is_blocksparse = True, 返回 BlocksparseFlashAttentionBackend
2. 如果 is_attention_free, 返回 PlaceholderAttentionBackend, 这个类什么也不做

3. global_forced_backend 全局指定, None
4. 从指定的环境变量读取
5. 一般走到这里: 根据平台选择 attn_backend
current_platform.get_attn_backend_cls(
        selected_backend, head_size, dtype, kv_cache_dtype, block_size, use_v1,
        use_mla)
        第一个参数 selected_backend 是通过环境变量指定的 cls

Platform
    CpuPlatform
    HpuPlatform
    CudaPlatformBase
        NvmlCudaPlatform
        NonNvmlCudaPlatform
    TpuPlatform
    ...


对于 CudaPlatform
    get_attn_backend_cls(...)
        在这里通过各种参数, 选择 Backend
        if use_mla:
            满足特定条件1 return "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
            满足特定条件2 return xxx
        if use_v1:
            ...

```
