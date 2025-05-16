# 使用 python 写 核函数
# 使用 Numba 编写 CUDA 核函数（推荐）
# numba 是一个 JIT 编译器库，支持用 Python 写 CUDA 核函数并直接在 GPU 上运行。

from numba import cuda
import numpy as np


@cuda.jit
def add_one_kernel_2d(arr):
    row, col = cuda.grid(2)  # 获取grid中的坐标
    if row < arr.shape[0] and col < arr.shape[1]:
        arr[row, col] += 100  #


def _test_2d():
    row, col = 5, 7
    arr = np.arange(row * col, dtype=np.int32).reshape(row, col)

    # cop to gpu
    d_arr = cuda.to_device(arr)

    # 启动核函数, 1个block * 10线程
    threads_per_block = (16, 16)
    blocks_per_grid_x = (arr.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (arr.shape[1] + threads_per_block[0] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    add_one_kernel_2d[blocks_per_grid, threads_per_block](d_arr)

    # copy back to host
    result = d_arr.copy_to_host()
    print(result)


if __name__ == "__main__":
    _test_2d()
