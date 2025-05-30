# 问题 
考虑一个 seq ，如果不 chunk，处理流程是这样的（我的理解）：
1.  进行 embed, 得到 (1, N, d) 的tensor
2.  进行 positional embed, 仍然是 (1, N, d)
3. 然后进入 attention层
计算 qkv, 然后计算 attn score, 计算 output = score * v
然后是 ffn, 通常是 d*4d, 4d*d 2个线性层
attention层的输出是 (1, N, d)
attention有多层
4. 到 lm_head + softmax, 得到 logits = (1, N, log_probs)
logits[:,-1,:] 就是下一个token的 概率

如果 chunked prefill, 从哪个阶段开始 chunk? 具体是怎样操作的呢? 

## chunked prefill 从哪个阶段开始分块？
chunk 是从 Transformer 的前向传播阶段开始的，而不是 embedding 阶段。
不在 embedding 阶段 chunk 的原因：
- embedding 是一个非常轻量的操作，占用资源非常少
- positional embedding 是依赖 token index，也很轻量级
- 而 attention 和 FFN 是计算和内存最重的部分，所以 chunk 就从这里开始。

## chunked prefill 的实现逻辑
我们假设输入序列 token 长度为 N，嵌入维度为 d，chunk 大小为 C（例如 128）：

### Step 1: 先处理 embedding
input_ids → token embedding → (1, N, d)
          → add positional embedding → (1, N, d)

### Step 2: Transformer 分块处理
- 我们将 [0, N) 分成多个长度为 C 的 chunks：
```
chunk 1: tokens [0, C)
chunk 2: tokens [C, 2C)
chunk k: tokens [k*C, N)
```
- 对于每个 chunk，执行一次 forward


# 为什么需要 chunked prefill？
在大模型推理（特别是 prefill 阶段，即输入 prompt 时中，处理长序列会遇到两个瓶颈：
- 显存不够（KV cache 太大）
- 整体GPU资源利用率低
因此，chunked prefill 的目标是 将 prompt 分块处理，减少一次性计算和内存占用，并可以更好地并行化。



# refer
- https://freedium.cfd/https://donmoon.medium.com/llm-inference-optimizations-2-chunked-prefill-764407b3a67a
