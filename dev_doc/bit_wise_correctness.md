2023 meetup ppt中有这样一句:

Because of the custom CUDA kernels, vLLM cannot always ensure bit-wise correctness

# 原因 (gpt-4o)
bit-wise correctness 意味着：相同的输入，模型每次输出的每一位（bit）都是完全一致的。
这要求计算结果在数值上100%一致，哪怕是最后一位的小数也不变。

这在某些场景下非常重要，比如：
测试模型是否“回归”（regression tests）
部署对输出敏感的任务（如金融、科学计算）

## vLLM 为什么无法保证？
vLLM 之所以不能完全保证 bit-wise correctness，主要是因为它实现了 自定义 CUDA kernel（例如 PagedAttention） 来提高性能，
这可能导致下面几类非确定性行为：

1. 浮点数加法不具交换律
GPU 上进行的并行浮点加法顺序可能会变化（取决于线程调度），例如：
(a + b) + c ≠ a + (b + c) // 由于浮点精度限制
追问： 
部署时，只要开了 tensor parallel就不能保证 bit-wise correctness
部署时，只要开了 tp, 也不能保证相同输入，得到相同的 bit-wise。因为并行执行不保证顺序

2. 使用 atomic 操作或 warp shuffle
PagedAttention 等 kernel 可能使用了 __shfl_sync 或 atomicAdd 等 CUDA 原语，这些本身可能是 非确定性的。

3. 优化级别不同导致编译器行为变化
自定义 kernel 编译时使用的 -O3 等优化可能会引入浮点运算重排序，也影响精度。

## 如果你真的需要 bit-wise 正确：
你可能要回退到 HuggingFace Transformers + PyTorch 的 FP32 模式，不用任何 kernel fusion。
或者关闭所有优化，比如 FlashAttention、TensorRT、vLLM 等。
