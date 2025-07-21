# 问题1. 我想创建一个 conda环境, 然后从源码安装 vllm.  选择哪个版本的python？
建议使用 Python 3.10。
原因：
vllm目前（截至2025年）对Python 3.10支持非常好。
兼容性最佳（比如PyTorch、CUDA库等也对3.10优化良好）。
一些新特性（比如pattern matching）也能用，但又不会因为太新导致第三方库兼容问题（比如3.12有些库可能还没完全适配）。

```bash
conda create -n vllm_dev python=3.10 -y
conda activate vllm_dev
```

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
    - 这是一个 conda包, 也就是说不能用 pip安装

# 问题4. 检查安装vllm环境

```
python -c "import torch; print(torch.version.cuda);"
```

```python
import torch

print(torch.__version__)  # 应该是2.x版本，比如2.1.x或2.2.x
print(torch.version.cuda)  # 应该显示 '11.8'
print(torch.cuda.is_available())  # 应该是 True
```

# 问题5. 关于nvcc

```bash
which nvcc
/usr/local/cuda-12.4/bin/nvcc
# 注意这个nvcc路径是系统全局 cuda-12.4的, 不是conda环境中的(因为未安装 cudatoolkit)
# 是否有影响 vllm的源码编译
```

vllm编译时主要依赖的是：
PyTorch 的 CUDA runtime（你的conda环境里已经有了）
系统gcc/g++编译器
（可选）如果需要自己编译 CUDA kernel，才会查找 nvcc
但是 vllm源码用的是PyTorch Extension机制，默认优先用 PyTorch 自带的环境配置，不强制要求环境里有nvcc。
vllm源码编译时 -> 主要看 PyTorch能否正确提供环境，不直接用系统nvcc。

# 源码安装

```bash
uv pip install -r requirements/build.txt
export MAX_JOBS=16
VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e .
# 编译有点耗时

# 这里还有一些包要安装
uv pip install -r requirements/common.txt
uv pip install -r requirements/dev.txt
```

- VLLM_USE_PRECOMPILED=1: 直接使用vllm官方编译好的二进制扩展（wheel包里的.so文件），不自己编译C++/CUDA代码
- 这种是 python-only build
- 如果哪天修改了 c++/cuda代码, 则去掉这个重新编译
- 安装日志: 可以看到安装过程卸载了 pytorch, 并重新安装了
