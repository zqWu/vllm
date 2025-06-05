# 假设
- Wq, Wk, Wv 都是 dim * dim
- 有 n_layer层 多头注意力层
- 数据类型dtype_size占据的字节数(fp16=2字节, fp32=4字节) 

# 单层attn 计算需要的 kv cache数量
- n_head * head_dim = dim
- qkv的形状:
  - q: h_head个, shape = 1 * head_dim, 简单描述为 n_head * head_dim
  - k, v相同
- 一个token对应的 kv cache, 其内存/显存占用大小
  - k: n_head * head_dim * dtype_size. 注意这是单层中的k
  - v: 与k一样

## 注:
实际kv cache是一个 3d tensor
k: (n_layer, n_head, seq_len, head_dim)
v: (n_layer, n_head, seq_len, head_dim)

## 总计kv cache
 - 2 * n_layer * n_head * head_dim * dtype_size * token数量
 - = 2 * n_layer * dim * dtype_size * token数量

## 示例
dim = 4096
n_layer=32
dtype = fp16
那 1000个token, kv cache = 2 * 32 * 4k * 1k * 2字节 = 512M

# 遗留问题
- kv cache的shape为什么是 (n_layer, n_head, seq_len, head_dim)
- kv cache的 读写时机 和使用
