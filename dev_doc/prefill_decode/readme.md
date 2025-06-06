# 一直以来我搞不清 为什么有 prefill 与 decode的区别

##

| 项目            | Prefill          | Decode                         |
|---------------|------------------|--------------------------------|
| 输入 token 数    | 多个（完整 prompt）    | 通常一个                           |
| 是否使用 KV Cache | 否（初始构建 KV Cache） | 是（复用历史 KV）                     |
| Attention 类型  | Full Attention   | Causal Attention with KV Cache |
| 典型用途          | 初始上下文处理          | 生成阶段的每一步                       |
| 计算开销          | 较高               | 极低（非常高效）                       |

## 如果不考虑 kv cache, prefill 与 decode有区别吗

Prefill 和 Decode 本质上没有本质差别，都是通过 Transformer 模型计算 token 的下一个概率分布。

## 用我自己的话来讲 pd的差别

prefill阶段, 假设输入=3 token, dim=100, 在每个attn layer, 要做这样的事:
    计算 qkv, 都是 3* 100
    计算 每个token对应的 输出 softmax(q@k.T/factor) dot v, 输出维度 = 3 x 100
    然后 这个 3x100作为下一个 attn层的输入

decode阶段, 假设输入=3 token + 1token(新生成的), 在每个 attn layer, 要做这样的事
    计算 new token的 qkv, 都是 1*100, 其他3个token的qkv 使用 kv cache
    计算 new token对应的输出, softmax(q_new @(k_old + k_new).T) dot (v_old+v_new)
                                    1 x 100 @ 100 * 4              4 * 100
        得到 1 * 100
   然后这个 1*100作为下一层的输入
