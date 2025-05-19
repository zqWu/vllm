# 术语
FMA fused multiply-add
FFMA fused floating-point multiply-add
cuda中 FMA和FFMA是一个内容, 表达不同

HBM = high bandwidth memory 高带宽内存

https://www.youtube.com/watch?v=gBMO1JZav44&t=683s

# 原始矩阵乘法计算量
shape(A) = m * k
shape(B) = k * n

那么 shape(A * B) = m * n
要计算 m * n 个元素, 每个元素有 k个 (相乘再相加, cuda中 FMA) 操作

总的计算量 ~ m * n * k

# 单个 朴素 attention 中的计算
假设 Wq, Wk, Wv 都是 d_model * d_model
假设 输入X X.shape = N * d_model. N个token, 每个token embd = dim


q = X @ Wq, shape = N * d_model
k, v 与 q 形状一样

q @k.T 需要进行  N * N * d_model次 FMA.
得到的结果, shape(q @ k.T) = N * N

## 朴素 attention中, 计算量 ~O(N^2), memory ~O(N^2)


# flash attn
https://www.youtube.com/watch?v=gBMO1JZav44


## 核心问题
- 显存占用过大 (q @ k.T, softmax @ v), 特别是长 seq时, 显存消耗很大
- 瓶颈不在 算力, 而在于读写上.
  - 着重降低对显存数据的访问次数

## 核心思想
- tiling
- fused kernel

# A800-80G
- SRAM (L2 cache) = 80M
- 模型: qwen2.5, token embed dim = 5120
  - 单个token, fp16的情况下
    - qkv 矩阵, shape = 1 * 5120
    - 数据量 = 1 * 5120 * 2bytes = 10k
    - 单个token, qkv 一共有 30k

## 简单问答 prompt 对应的token < 1024
- qkv 一共占 30k * 1024 = 30 M, 完全可以存放在 sram 中

## 长上下文, token数量 = 4k
- qkv 一共占 30k * (4k) = 120M, 无法完全存放到 sram中

## 更长的上下文, token = 16k
- q的存储 = 10k * 16k = 160M, q已经不能存放到 sram中
