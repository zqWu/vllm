# Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
vllm 项目运行时有这个日志。GPU是 GTX-1660 TI. 
问题: 这种不支持是 软件实现层面未实现，还是硬件结构不支持 导致的？

FlashAttention 的高性能实现依赖于：

Tensor Cores 的 BF16 / FP16 高速矩阵乘法支持
Shared memory / warp-level primitives 的高效访问模式
较高的 memory bandwidth 和较低的 latency

而这些优化：
在 Ampere (RTX 30 系列) 及更新架构（如 Ada, Hopper）中，才具备 完整支持
在 Turing（GTX 16xx / RTX 20 系列）或 Volta 上，
硬件资源（如 shared memory 结构、Tensor Core 能力）达不到 FlashAttention v2 所需的性能要求，
运行反而会变慢，甚至出错


# Using XFormers backend. 
是不是就是传统的 transformer 方式计算 attention?

xFormers 是 Meta AI 提供的高性能 Transformer 模块库，
其中包含多种 高效注意力（efficient attention）实现，其目标是：

减少显存占用（Memory Efficient Attention）
在不支持 FlashAttention 的硬件上提供替代方案
尽可能加速标准 Transformer 的 Attention 运算


xFormers 的注意力实现特点：
使用 MemoryEfficientAttention（不是标准 softmax attention）
核心思想：不构造完整的 attention matrix（比如 QK^T），而是直接计算所需的输出，减少内存和计算量
计算方式经过 CUDA 优化，但和 FlashAttention 相比还是慢一些（不过比 PyTorch 原生实现快）
