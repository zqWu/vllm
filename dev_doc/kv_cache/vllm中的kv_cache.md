# opt-125m模型信息

- config.json

```json
{
  "_name_or_path": "facebook/opt-125m",
  "activation_dropout": 0.0,
  "activation_function": "relu",
  "architectures": [ "OPTForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 2,
  "do_layer_norm_before": true,
  "dropout": 0.1,
  "eos_token_id": 2,
  "ffn_dim": 3072,
  "hidden_size": 768,
  "init_std": 0.02,
  "layerdrop": 0.0,
  "max_position_embeddings": 2048,
  "model_type": "opt",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "prefix": "</s>",
  "torch_dtype": "float16",
  "transformers_version": "4.21.0.dev0",
  "use_cache": true,
  "vocab_size": 50272,
  "word_embed_proj_dim": 768
}
```

- opt-125m使用标准的多头注意力机制
    - num_attention_heads = 12
    - hidden_size = 所有头的总维度 = 768
    - 每个头的维度(head_dim) = hidden_size/num_attention_heads = 64

# 什么是 kv cache

- k/v是注意力计算过程的tensor.
- k/v 可复用, 因此进行了cache, 这就是kv cache
- kv cache是一个多维 tensor

# kv是怎么运算得到的

## 第一个 attn层之前

- N个token序列 [t1, ... tN-1, tN] = (N,1)形状的 tensor
- attention 计算过程中, 注意力层的模型权重包含 Wq, Wk, Wv, 一般是shape=(dim, dim)
  dim 就是 配置中的hidden_size, 在 opt-125m模型中 dim=768.
- 经过 embedding + pe 之后, 得到 tensor, 形状 (1, N, dim) = (batch, token数, dim)

## 第一个attn层
- 输入tensor = (1, N, dim)
- 计算多头qkv, 以q为例(k/v是一样的逻辑)
- 12个头, embed @ Wk 得到的就是 k, 形状=(1, N, n_head, head_dim)
- 得到 qkv后, 计算 attention + softmax + layer_norm + ffn, 得到tensor = (1, N, dim)
- attn层的输入与输出tensor, 是相同的形状 = (B, N, D)

## 每一层都有 kv cache
- 每一个token, 在每一层, 都会得到独立的k/v
- opt-125m 有12层 num_hidden_layers
- 假设输入 tokens = [t1, t2]
- 会生成这些 kv

```
layer0_t1_k
layer0_t1_v
layer0_t2_k
layer0_t2_v
...
layer11_t1_k
layer11_t1_v
layer11_t2_k
layer11_t2_v
```

# 既然是张量(tensor), 那这个 k 是怎样的形状
- 取决于模型的 attention的实现
- 以下是 flash-attn 和 paged-attn中, kv-cache这个 tensor的shape

# 既然是 cache, 这个 cache是如何分配内存存储的, 又是怎么进行读取使用的

## 存储
- kv_cache自从分配了内存后, 就使用 block_manager进行管理
- k_cache里面, 每个block都能存放若干个key(默认16), 同理 v_cache
- k/v 的计算是在模型的 attn层计算得到的tensor, 但是存储位置是在 vllm调度时就指定的
- 调度的结果: token_3 对应的key, 都存放在 block=1, slot=10的位置
  - token_3 在第1层的key, 存放到 k_cache[第1层][block=1][slot=10]
  - token_3 在第1层的val, 存放到 v_cache[第1层][block=1][slot=10]
  - token_3 在第2层的key, 存放到 k_cache[第2层][block=1][slot=10]
  - token_3 在第2层的key, 存放到 k_cache[第2层][block=1][slot=10]
  - ... 一直到第12层
- 存储时机: 在计算完q,k,v时, 就进行存储

## 读取

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

    def forward(self, ..., kv_cache: torch.Tensor, ):
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
    k=key_cache, <= == == == == == == == == == == 整块内存
v = value_cache,
out = output[:num_actual_tokens],
cu_seqlens_q = cu_seqlens_q,
max_seqlen_q = max_seqlen_q,
seqused_k = seqused_k, <= == == == == == == seq长度, 如[19, 8]
max_seqlen_k = max_seqlen_k,
softmax_scale = self.scale,
causal = True,
alibi_slopes = self.alibi_slopes,
window_size = self.sliding_window,
block_table = block_table, <= == == == == 整块内存中使用了哪些
block, 如[[2, 3, 0], [4, 0, 0]]
softcap = self.logits_soft_cap,
scheduler_metadata = attn_metadata.scheduler_metadata,
fa_version = self.vllm_flash_attn_version,
q_descale = layer._q_scale.expand(descale_shape),
k_descale = layer._k_scale.expand(descale_shape),
v_descale = layer._v_scale.expand(descale_shape),
)
```
