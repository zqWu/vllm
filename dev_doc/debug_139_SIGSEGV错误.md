我使用 pycharm debug  vllm （使用 ssh remote ）, 程序 run 能够正常，
但是debug 会报错：Process finished with exit code 139 (interrupted by signal 11:SIGSEGV)

通常这个是什么原因导致？

# 解决 
export PYDEVD_USE_FRAME_EVAL=NO

# 原因
PYDEVD_USE_FRAME_EVAL=YES（默认）时，PyCharm 调试器会 插桩（instrument）Python 字节码执行过程，以便更高效地设置断点、单步调试。
PYDEVD_USE_FRAME_EVAL=NO 时，PyCharm 回退到传统的 sys.settrace() 调试方式，更慢但更兼容。


## 与 C 扩展不兼容（如 PyTorch、vLLM、Triton、CUDA 等）
Frame-eval 会修改 Python 字节码执行路径，对于依赖 PyTorch JIT、
自定义 C++ op 的框架（比如 vLLM、FlashAttention、Triton kernel）来说，这会破坏原本的调用栈或内存布局。
尤其是在涉及 GPU memory management 和多线程的时候，会触发非法访问，导致 SIGSEGV。

## 与多线程或 fork 进程不兼容
frame-eval 在处理多线程调度或 fork 子进程（常见于 vLLM 服务器启动过程）时，容易出现 race condition。
你可能不会在“run”模式下遇到这些问题，是因为 debugger 本身的字节码注入导致行为变化。