#include <iostream>
#include <cuda_runtime.h>

// 核函数：每个线程将数组对应元素加一
__global__ void addOne(int* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] += 10;
    }
}

int main() {
    const int size = 10;
    int h_data[size];  // host 上的数据

    // 初始化数据
    for (int i = 0; i < size; ++i) {
        h_data[i] = i;
    }

    int* d_data;
    cudaMalloc(&d_data, size * sizeof(int));  // 在 GPU 上分配内存
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);  // 复制数据到 GPU

    // 每个线程处理一个元素，1 个 block 10 个线程
    addOne<<<1, size>>>(d_data, size);

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);  // 拷贝结果回 host
    cudaFree(d_data);  // 释放 GPU 内存

    // 打印结果
    for (int i = 0; i < size; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

// 编译: nvcc -o add_one add_one.cu
// 执行: ./add_one