# 从源码安装

- 首先检查 机器 `nvidia-smi` 得到 cuda=12.4, 也就是说我的cudatoolkit 最高版本可以=12.4
- 看一下 requirements/cuda.txt 中, 默认是 torch==2.8.0
- 去pytorch 官网看到 2.8.0时, cuda12.6, 超过了机器驱动版本
- 因此选择 torch=2.7.1 + cuda=11.8 + python=3.12 (3.10也ok)


```bash
# 整个过程中无需 conda
uv venv .venv --python=3.12 --seed
source .venv/bin/activate
# conda create -n vllm_dev python=3.12 -y
# conda activate vllm_dev

# pytorch
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.version.cuda);" # 检查安装


# 使用指定torch. 这会修改一些文件, 安装后需要恢复
python use_existing_torch.py

uv pip install -r requirements/build.txt
uv pip install -r requirements/common.txt

# 编译+安装, 耗时, 需要github

# 配置github加速
export https_proxy=http://127.0.0.1:7890 && export http_proxy=http://127.0.0.1:7890
export MAX_JOBS=64
pip install --no-build-isolation -e . --verbose --no-deps

# 因为非默认版 torch, 需要build .so, 否则运行报错 xxx.so: undefined symbol
# VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e .
# 等待成功即可


# 最后 checkout 修改过的 文件, 这些是 `python use_existing_torch.py` 导致的
git checkout requirements/*.txt pyproject.toml
```

- VLLM_USE_PRECOMPILED=1: 直接使用vllm官方编译好的二进制扩展（wheel包里的.so文件），不自己编译C++/CUDA代码
- 这种是 python-only build
- 如果哪天修改了 c++/cuda代码, 则去掉这个重新编译
- 安装日志: 可以看到安装过程卸载了 pytorch, 并重新安装了


# 问题2. 源码安装 vllm 需要先配置 cuda和pytorch吗？
- 不需要, 自动安装 (版本根据vllm的依赖决定)
- 也可以安装指定的 torch, 然后源码编译安装

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# 注意这里安装了 pytorch-cuda=11.8
```

# 问题3. 需要安装 cudatoolkit==11.8吗？
- 不需要, 自动安装
- 不推荐手动动装 cudatoolkit==11.8
pytorch-cuda=11.8 它会从 nvidia 官方源安装适配 PyTorch 的 CUDA 11.8 runtime
（包括 cudnn、driver stub、nvcc 兼容库等）。这是推荐方式。

pytorch-cuda 和 cudatoolkit 会冲突，因为它们都提供了 libcudart.so、libcublas.so 等文件。
如果你用的是 pytorch-cuda=11.8（从 -c nvidia），它已经把所需的 CUDA runtime、cudnn 等都带好了。
vLLM 也推荐这种方式，因为它更接近官方NVIDIA发布的库结构。

## 追问 cudatoolkit与 pytorch-cuda 有何异同
- cudatoolkit是 anaconda提供的cuda runtime+部分driver stub. 目的是通用cuda支持
- pytorch-cuda是nvidia官方为pytorch提供的 cuda runtime + cuDNN + 依赖包合计.
    - 大致相对于 cudatoolkit + cudnn + nccl 组合
    - 不是通用的cuda工具包, 而是专门为pytorch定制的
    - 这是一个 conda包, 不能用 pip安装

# 检查安装vllm环境

```
python -c "import torch; print(torch.version.cuda);"
```

```python
import torch

print(torch.__version__)            # 应该 = 2.7.1
print(torch.version.cuda)           # 应该 = 11.8
print(torch.cuda.is_available())    # 应该 = True
```

# 关于nvcc
```bash
which nvcc
/usr/local/cuda-12.4/bin/nvcc
# 这个nvcc路径是系统全局 cuda-12.4的, 不是conda环境中的(因为未安装 cudatoolkit)
# 是否有影响 vllm的源码编译, 特别是 .cu文件的编译?
```
- vllm编译时主要依赖的是：PyTorch 的 CUDA runtime（你的conda环境里已经有了）
- vllm源码编译时 -> 主要看 PyTorch能否正确提供环境，不直接用系统nvcc
- 后面测试了, 无需 cudatoolkit 或 pytorch-cuda=11.8, 可以编译 .so


问题: 为什么 vLLM 可以不用系统的 nvcc 也能编译 .cu 文件？

1. PyTorch 提供了自己的编译工具链封装
torch.utils.cpp_extension（以及内部的 CppExtension, CUDAExtension）会在构建时调用 PyTorch 提供的 cpp_extension.py。
它会自动帮你找出 CUDA include path / lib path，并且选择合适的编译器。

2. pytorch 自带了 CUDA headers（来自 pytorch-cuda 或 cudatoolkit 包）
当你安装 torch==X.Y.Z+cu118 这样的 wheel 时，里面已经包含了一整套 CUDA runtime 库 + headers，
例如 cuda_runtime_api.h、cuda.h、device_launch_parameters.h 等。
编译 .cu 文件时，并不是直接依赖 /usr/local/cuda 里的头文件，而是优先用 PyTorch wheel 内置的 headers。

3. 实际编译过程里，PyTorch 可以用 clang++ / nvcc / gcc 路径
如果你机器上没有 nvcc，PyTorch 会 fallback 到使用 clang 的 CUDA 前端（即 clang++ -x cuda）。
如果你机器上有 nvcc，它可能会用系统的 /usr/local/cuda/bin/nvcc，但 include 路径还是优先用 PyTorch wheel 里的 include。

4. 运行时不依赖 nvcc
.so 编译好之后，运行时只需要 CUDA runtime (libcudart.so 等)
所以即使系统没有 CUDA Toolkit，vLLM 依然可以编译和运行。
