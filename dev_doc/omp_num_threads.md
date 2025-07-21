OMP_NUM_THREADS 在 vLLM 中的作用
虽然 vLLM 的主推理框架是高度优化的 CUDA + KV Cache 流程，但还是会有部分操作可能触发 CPU 多线程运算。例如：

Tokenizer 的并发处理（如果在 CPU 上）

前后处理（logits 后处理、sampling）部分使用 CPU 运算

非 batch 推理阶段 CPU 执行时间可能相对多一些

vLLM + HuggingFace + transformers 中某些组件默认使用 OpenMP 并发

设置 OMP_NUM_THREADS 影响的是：在这些环节中，使用 CPU 进行 OpenMP 并行时，每个进程/线程最多使用几个线程并发运行。
