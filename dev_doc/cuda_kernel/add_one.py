# 使用 python 写 核函数
# 使用 Numba 编写 CUDA 核函数（推荐）
# numba 是一个 JIT 编译器库，支持用 Python 写 CUDA 核函数并直接在 GPU 上运行。

from numba import cuda
import numpy as np


@cuda.jit
def add_one_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.shape[0]:
        arr[idx] += 1


def _test_1d():
    n = 10
    arr = np.arange(n, dtype=np.int32)

    # cop to gpu
    d_arr = cuda.to_device(arr)

    # 启动核函数, 1个block * 10线程
    threads_per_block = 32
    block_per_grid = (n + threads_per_block - 1) // threads_per_block
    add_one_kernel[block_per_grid, threads_per_block](d_arr)

    # copy back to host
    result = d_arr.copy_to_host()
    print(result)


if __name__ == "__main__":
    _test_1d()
