# speculative sampling / decoding
推测解码
https://arxiv.org/abs/2302.01318#


# llm 组件、架构

vocab       这个是 token映射表, token --- token_id(整数)
tokenizer   把自然语言转换成 token
embed+pe    把 [token_id, ..., token_id] 映射到 [dim维向量], shape = (V, d)
pe          位置编码
decoder     attention层, (Batch, N_token, dim) -> (B,N,d)
lm_head     (d, V), 一般与 embed复用. (B,N,d) ------>(B,N,V)
            经过softmax后得到 (B,N,V)
            logits[:,-1,:] 表示下一个token的概率

## 对于 lm_head 输出的例子说明
以 opt-125m为例, 它的dim=768, Vocab=V=50272

Input tokens: [The, cat, sat, on] -> ids: [  10, 531, 872,  87]
模型 decoder 输出 shape： (Batch, N_tokens, dim) → (1, 4, 768)
lm_head 之后的 logits： (1, 4, 50272) → 每个位置预测的是“当前位置的下一个 token 的概率”
    logits[:, 0, :] → 预测的是 "The" 的下一个 token
    logits[:, 1, :] → 预测的是 "cat" 的下一个 token
    logits[:, 3, :] → 预测的是 "on" 的下一个词 → 我们关心的就是这个


# 关于 target model 推理 Mq(pf, x1, x2, x3, x4, x5)
- 之前理解为 需要 batch进行处理，可能是错误思路
- Mq(pf, x1, x2, ... ,x5), 只需要推理一次, 就能得到
  - q(x1), ..., q(x5)

# 算法关键
1. 使用 draft model 串行生成 N个token
2. 使用 target model 一次验证 N个token, 根据reject-sampling 从中accept 前k(0-N)个, 并sampling得到 第 k+1个token