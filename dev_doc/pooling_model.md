# 问题1: 关于 pooling model

vLLM 还支持池化模型，包括嵌入模型、重排序模型和奖励模型。

1. 什么是 池化模型 pooling model？
2. 为什么 嵌入模型、重排序模型和奖励模型 是 pooling model
3. 什么是 "pooling request to the LLM". 目前知道 completion request to the llm .
   下面是 vllm中的一个源码和注释

```python
class PoolingRequestOutput(Generic[_O]):
    """
    The output data of a pooling request to the LLM.
```

1. 什么是 池化模型 (pooling model)?

在 LLM 推理里，“池化 (pooling)”的意思是：
把序列中每个 token 的 hidden state 向量，压缩成一个 固定维度的单一向量。
这个操作通常不是生成文本，而是得到一个 representation 。

常见方式：
mean pooling（对所有 token embedding 求平均）、
CLS pooling（取特殊 token 向量）、
last token pooling（取最后一个 token 向量）等。

换句话说：
completion model：输入 prompt，输出下一个 token 概率 → 生成文本。
pooling model：输入 prompt，输出一个 pooled 向量 → 用于相似度计算、打分等。

2. 为什么嵌入模型、重排序模型和奖励模型是 pooling model？

嵌入模型 (embedding model)
目标：把文本转成向量（用于检索、语义搜索等）。
方式：对 hidden states 进行 pooling，得到单个 embedding。

重排序模型 (reranker model)
输入：query + candidate documents。
输出：每个 candidate 的相关性分数。
本质：对 query/document 各自做 pooling → 得到向量 → 计算匹配分数。

奖励模型 (reward model)
输入：生成的候选序列。
输出：一个数值奖励分数。
实现方式：通常对整段序列做 pooling，再通过一个 head 得到标量 reward。

共同点：它们的输出不是序列，而是 单个向量或标量，所以都属于 pooling model。


# 问题: 常见的 reranker / embedding 模型，属于 llm 模型的范围吗？

严格来说 reranker / embedding 模型 和传统意义上的 LLM (decoder-only Transformer) 并不完全一样，
但在 vLLM 框架里，它们一般也被统一称作 LLM family，原因如下：


1. 从架构角度看
Embedding 模型
本质：一个 Transformer encoder 或 decoder-encoder，输出每个 token 的 hidden states。
不做解码，只做 pooling → 得到固定向量。
例子：OpenAI text-embedding-3-small, BERT, E5 等。

Reranker 模型
多数是 cross-encoder Transformer（把 query + document 拼接输入 Transformer）。
输出 hidden states，经过 pooling + 打分头 (MLP) → 得到相关性分数。
例子：monoT5, bge-reranker, OpenAI 的 text-rerank 模型。

Reward 模型
通常也是 Transformer backbone（和 LLM 架构几乎一致）。
对生成文本的 hidden states 做 pooling，然后接 reward head → 得分。

可以看到，它们和 LLM 的区别不在于 backbone，而在于 输出 head 的设计。


2. 从 vLLM 的实现角度看
vLLM 的推理引擎支持两类：
Completion models → 典型的 decoder-only LLM，用 KV cache 做增量解码。
Pooling models → embedding/reranker/reward，输入一次，输出向量/分数，不需要逐步生成。
对 vLLM 来说，这些都可以走同一个推理后端（都是 Transformer forward），只是最后输出形式不同。


3. 所以，算不算 LLM？
如果 狭义定义 LLM = “能生成文本的大语言模型” →
embedding/reranker/reward 模型 不算 LLM，而是 Transformer 表征模型。

如果 广义定义 LLM = “大规模预训练的 Transformer 模型（无论是生成还是表征）” →
它们 属于 LLM 范畴。

在工程框架（比如 vLLM、OpenAI API）里，embedding / rerank / reward 模型 被归为 LLM 系统的子类，
因为它们共享相同的 backbone 和推理优化。

总结: 
embedding / reranker / reward 模型 ≠ 传统意义的 LLM（不会生成文本）
它们 通常基于相同的 LLM 架构（大规模 Transformer），所以在 vLLM 等框架里被纳入 LLM 家族，
只是对应的是 pooling model 而不是 completion model。

```text
                  ┌──────────────────────────────┐
                  │            vLLM              │
                  │    (统一推理后端: Transformer) │
                  └──────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
 ┌───────────────────────┐            ┌─────────────────────┐
 │   Completion Models   │            │    Pooling Models   │
 └───────────────────────┘            └─────────────────────┘
                │                                │
   ┌────────────┼────────────┐        ┌──────────┼───────────┐
   │            │            │        │          │           │
   ▼            ▼            ▼        ▼          ▼           ▼
Text Generation Chat Models  Code    Embedding   Reranker   Reward
(LM head)      (Chat head)   Models  Models      Models     Models
                                       │          │           │
                                       ▼          ▼           ▼
                                 Dense vector   Relevance   Scalar
                                representation   score     reward
```

# bge-m3 embedding 模型架构
- XLM-RoBERTa architecture的一个变种
```text
[Text] ──► [Tokenizer] ──► [Transformer Encoder] ──► Hidden states (per token)
```