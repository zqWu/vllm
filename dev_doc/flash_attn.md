# 术语
FMA fused multiply-add
FFMA fused floating-point multiply-add
cuda中 FMA和FFMA是一个内容, 表达不同

HBM = high bandwidth memory 高带宽内存

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
- 瓶颈不在 算力, 而在于读写上
- 着重降低对显存数据的访问次数
