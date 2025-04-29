# 问题1. 我想创建一个 conda环境, 然后从源码安装 vllm.  选择哪个版本的python？
建议使用 Python 3.10。
原因：
vllm目前（截至2025年）对Python 3.10支持非常好。
兼容性最佳（比如PyTorch、CUDA库等也对3.10优化良好）。
一些新特性（比如pattern matching）也能用，但又不会因为太新导致第三方库兼容问题（比如3.12有些库可能还没完全适配）。

```bash
conda create -n vllm-dev python=3.10 -y
conda activate vllm-dev
```

# 问题2. 源码安装 vllm 需要先配置 cuda和pytorch吗？ 更新: 不需要, 自动安装

是的，安装vllm源码之前，应该先装好：
PyTorch（而且要对应正确的 CUDA 版本）
CUDA 驱动（系统里的 driver + runtime 库）
vllm 的源码编译需要用到 PyTorch 的 C++/CUDA extension（比如它里面的vllm._C模块）。
vllm 也需要根据你PyTorch是CUDA版还是CPU版，决定怎么编译和链接底层代码。
如果PyTorch还没装好，pip install -e .（源码安装）时会报错，比如 torch not found、nvcc not found、或者 C++/CUDA编译失败。

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# 注意这里安装了 pytorch-cuda=11.8
```

# 问题3. 需要安装 cudatoolkit==11.8吗？ 更新: 不需要, 自动安装
不要手动装 cudatoolkit==11.8
pytorch-cuda=11.8 它会从 nvidia 官方源安装适配 PyTorch 的 CUDA 11.8 runtime
（包括 cudnn、driver stub、nvcc 兼容库等）。这是推荐方式。

pytorch-cuda 和 cudatoolkit 会冲突，因为它们都提供了 libcudart.so、libcublas.so 等文件。
如果你用的是 pytorch-cuda=11.8（从 -c nvidia），它已经把所需的 CUDA runtime、cudnn 等都带好了。
vLLM 也推荐这种方式，因为它更接近官方NVIDIA发布的库结构。


# 问题4. 检查安装vllm环境
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
export MAX_JOBS=16
VLLM_USE_PRECOMPILED=1 pip install --editable .
# 编译有点耗时
```
- VLLM_USE_PRECOMPILED=1: 直接使用vllm官方编译好的二进制扩展（wheel包里的.so文件），不自己编译C++/CUDA代码
- 这种是 python-only build
- 如果哪天修改了 c++/cuda代码, 则去掉这个重新编译

# 安装日志
- 可以看到安装过程卸载了 pytorch, 并重新安装了

```bash
(vllm_dev) wuzhongqin@:~/github/vllm$ export MAX_JOBS=16
(vllm_dev) wuzhongqin@:~/github/vllm$ VLLM_USE_PRECOMPILED=1 pip install --editable .
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple, https://pypi.org/simple
Obtaining file:///data/project/wuzhongqin/github/vllm
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Collecting cachetools (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading cachetools-5.5.2-py3-none-any.whl.metadata (5.4 kB)
Collecting psutil (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached psutil-7.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (22 kB)
Collecting sentencepiece (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
Requirement already satisfied: numpy in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from vllm==0.1.dev6130+g506475d.precompiled) (2.2.5)
Requirement already satisfied: requests>=2.26.0 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from vllm==0.1.dev6130+g506475d.precompiled) (2.32.3)
Collecting tqdm (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting blake3 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading blake3-1.0.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.2 kB)
Collecting py-cpuinfo (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
Collecting transformers>=4.51.1 (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached transformers-4.51.3-py3-none-any.whl.metadata (38 kB)
Collecting huggingface-hub>=0.30.0 (from huggingface-hub[hf_xet]>=0.30.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached huggingface_hub-0.30.2-py3-none-any.whl.metadata (13 kB)
Collecting tokenizers>=0.21.1 (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached tokenizers-0.21.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting protobuf (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached protobuf-6.30.2-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)
Collecting fastapi>=0.115.0 (from fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading fastapi-0.115.12-py3-none-any.whl.metadata (27 kB)
Collecting aiohttp (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading aiohttp-3.11.18-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
Collecting openai>=1.52.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading openai-1.76.0-py3-none-any.whl.metadata (25 kB)
Collecting pydantic>=2.9 (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached pydantic-2.11.3-py3-none-any.whl.metadata (65 kB)
Collecting prometheus_client>=0.18.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading prometheus_client-0.21.1-py3-none-any.whl.metadata (1.8 kB)
Requirement already satisfied: pillow in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from vllm==0.1.dev6130+g506475d.precompiled) (11.1.0)
Collecting prometheus-fastapi-instrumentator>=7.0.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading prometheus_fastapi_instrumentator-7.1.0-py3-none-any.whl.metadata (13 kB)
Collecting tiktoken>=0.6.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading tiktoken-0.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
Collecting lm-format-enforcer<0.11,>=0.10.11 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading lm_format_enforcer-0.10.11-py3-none-any.whl.metadata (17 kB)
Collecting llguidance<0.8.0,>=0.7.9 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading llguidance-0.7.19-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.6 kB)
Collecting outlines==0.1.11 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading outlines-0.1.11-py3-none-any.whl.metadata (17 kB)
Collecting lark==1.2.2 (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached lark-1.2.2-py3-none-any.whl.metadata (1.8 kB)
Collecting xgrammar==0.1.18 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading xgrammar-0.1.18-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.6 kB)
Requirement already satisfied: typing_extensions>=4.10 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from vllm==0.1.dev6130+g506475d.precompiled) (4.13.2)
Requirement already satisfied: filelock>=3.16.1 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from vllm==0.1.dev6130+g506475d.precompiled) (3.18.0)
Collecting partial-json-parser (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading partial_json_parser-0.2.1.1.post5-py3-none-any.whl.metadata (6.1 kB)
Collecting pyzmq>=25.0.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading pyzmq-26.4.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (6.0 kB)
Collecting msgspec (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading msgspec-0.19.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
Collecting gguf>=0.13.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading gguf-0.16.2-py3-none-any.whl.metadata (4.4 kB)
Collecting importlib_metadata (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading importlib_metadata-8.7.0-py3-none-any.whl.metadata (4.8 kB)
Collecting mistral_common>=1.5.4 (from mistral_common[opencv]>=1.5.4->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading mistral_common-1.5.4-py3-none-any.whl.metadata (4.5 kB)
Collecting opencv-python-headless>=4.11.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
Requirement already satisfied: pyyaml in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from vllm==0.1.dev6130+g506475d.precompiled) (6.0.2)
Collecting einops (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached einops-0.8.1-py3-none-any.whl.metadata (13 kB)
Collecting compressed-tensors==0.9.3 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading compressed_tensors-0.9.3-py3-none-any.whl.metadata (7.0 kB)
Collecting depyf==0.18.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading depyf-0.18.0-py3-none-any.whl.metadata (7.1 kB)
Collecting cloudpickle (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading cloudpickle-3.1.1-py3-none-any.whl.metadata (7.1 kB)
Collecting watchfiles (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading watchfiles-1.0.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
Collecting python-json-logger (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading python_json_logger-3.3.0-py3-none-any.whl.metadata (4.0 kB)
Collecting scipy (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading scipy-1.15.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Collecting ninja (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (5.0 kB)
Collecting opentelemetry-sdk<1.27.0,>=1.26.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_sdk-1.26.0-py3-none-any.whl.metadata (1.5 kB)
Collecting opentelemetry-api<1.27.0,>=1.26.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_api-1.26.0-py3-none-any.whl.metadata (1.4 kB)
Collecting opentelemetry-exporter-otlp<1.27.0,>=1.26.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_exporter_otlp-1.26.0-py3-none-any.whl.metadata (2.3 kB)
Collecting opentelemetry-semantic-conventions-ai<0.5.0,>=0.4.1 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_semantic_conventions_ai-0.4.3-py3-none-any.whl.metadata (1.2 kB)
Collecting numba==0.61.2 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading numba-0.61.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.8 kB)
Collecting ray!=2.44.*,>=2.43.0 (from ray[cgraph]!=2.44.*,>=2.43.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading ray-2.43.0-cp310-cp310-manylinux2014_x86_64.whl.metadata (19 kB)
Collecting torch==2.6.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Using cached torch-2.6.0-cp310-cp310-manylinux1_x86_64.whl.metadata (28 kB)
Collecting torchaudio==2.6.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading torchaudio-2.6.0-cp310-cp310-manylinux1_x86_64.whl.metadata (6.6 kB)
Collecting torchvision==0.21.0 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading torchvision-0.21.0-cp310-cp310-manylinux1_x86_64.whl.metadata (6.1 kB)
Collecting xformers==0.0.29.post2 (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading xformers-0.0.29.post2-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (1.0 kB)
Collecting astor (from depyf==0.18.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading astor-0.8.1-py2.py3-none-any.whl.metadata (4.2 kB)
Collecting dill (from depyf==0.18.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading dill-0.4.0-py3-none-any.whl.metadata (10 kB)
Collecting llvmlite<0.45,>=0.44.0dev0 (from numba==0.61.2->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading llvmlite-0.44.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.8 kB)
Collecting interegular (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached interegular-0.3.3-py37-none-any.whl.metadata (3.0 kB)
Requirement already satisfied: jinja2 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled) (3.1.6)
Collecting nest_asyncio (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)
Collecting diskcache (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached diskcache-5.6.3-py3-none-any.whl.metadata (20 kB)
Collecting referencing (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading referencing-0.36.2-py3-none-any.whl.metadata (2.8 kB)
Collecting jsonschema (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached jsonschema-4.23.0-py3-none-any.whl.metadata (7.9 kB)
Collecting pycountry (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached pycountry-24.6.1-py3-none-any.whl.metadata (12 kB)
Collecting airportsdata (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading airportsdata-20250224-py3-none-any.whl.metadata (9.0 kB)
Collecting outlines_core==0.1.26 (from outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading outlines_core-0.1.26-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
Requirement already satisfied: networkx in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled) (3.4.2)
Collecting fsspec (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached fsspec-2025.3.2-py3-none-any.whl.metadata (11 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.4.127 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cuda-runtime-cu12==12.4.127 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cuda-cupti-cu12==12.4.127 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cublas-cu12==12.4.5.8 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cufft-cu12==11.2.1.3 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-curand-cu12==10.3.5.147 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cusparselt-cu12==0.6.2 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting nvidia-nccl-cu12==2.21.5 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvtx-cu12==12.4.127 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.7 kB)
Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Collecting triton==3.2.0 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached triton-3.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.4 kB)
Collecting sympy==1.13.1 (from torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached sympy-1.13.1-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from sympy==1.13.1->torch==2.6.0->vllm==0.1.dev6130+g506475d.precompiled) (1.3.0)
Collecting packaging (from lm-format-enforcer<0.11,>=0.10.11->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting deprecated>=1.2.6 (from opentelemetry-api<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading Deprecated-1.2.18-py2.py3-none-any.whl.metadata (5.7 kB)
Collecting importlib_metadata (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading importlib_metadata-8.0.0-py3-none-any.whl.metadata (4.6 kB)
Collecting zipp>=0.5 (from importlib_metadata->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached zipp-3.21.0-py3-none-any.whl.metadata (3.7 kB)
Collecting opentelemetry-exporter-otlp-proto-grpc==1.26.0 (from opentelemetry-exporter-otlp<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_exporter_otlp_proto_grpc-1.26.0-py3-none-any.whl.metadata (2.3 kB)
Collecting opentelemetry-exporter-otlp-proto-http==1.26.0 (from opentelemetry-exporter-otlp<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_exporter_otlp_proto_http-1.26.0-py3-none-any.whl.metadata (2.3 kB)
Collecting googleapis-common-protos~=1.52 (from opentelemetry-exporter-otlp-proto-grpc==1.26.0->opentelemetry-exporter-otlp<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading googleapis_common_protos-1.70.0-py3-none-any.whl.metadata (9.3 kB)
Collecting grpcio<2.0.0,>=1.0.0 (from opentelemetry-exporter-otlp-proto-grpc==1.26.0->opentelemetry-exporter-otlp<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached grpcio-1.71.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
Collecting opentelemetry-exporter-otlp-proto-common==1.26.0 (from opentelemetry-exporter-otlp-proto-grpc==1.26.0->opentelemetry-exporter-otlp<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_exporter_otlp_proto_common-1.26.0-py3-none-any.whl.metadata (1.8 kB)
Collecting opentelemetry-proto==1.26.0 (from opentelemetry-exporter-otlp-proto-grpc==1.26.0->opentelemetry-exporter-otlp<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_proto-1.26.0-py3-none-any.whl.metadata (2.3 kB)
Collecting protobuf (from vllm==0.1.dev6130+g506475d.precompiled)
  Downloading protobuf-4.25.7-cp37-abi3-manylinux2014_x86_64.whl.metadata (541 bytes)
Collecting opentelemetry-semantic-conventions==0.47b0 (from opentelemetry-sdk<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading opentelemetry_semantic_conventions-0.47b0-py3-none-any.whl.metadata (2.4 kB)
Requirement already satisfied: charset_normalizer<4,>=2 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from requests>=2.26.0->vllm==0.1.dev6130+g506475d.precompiled) (3.4.1)
Requirement already satisfied: idna<4,>=2.5 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from requests>=2.26.0->vllm==0.1.dev6130+g506475d.precompiled) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from requests>=2.26.0->vllm==0.1.dev6130+g506475d.precompiled) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from requests>=2.26.0->vllm==0.1.dev6130+g506475d.precompiled) (2025.1.31)
Collecting wrapt<2,>=1.10 (from deprecated>=1.2.6->opentelemetry-api<1.27.0,>=1.26.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached wrapt-1.17.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.4 kB)
Collecting starlette<0.47.0,>=0.40.0 (from fastapi>=0.115.0->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading starlette-0.46.2-py3-none-any.whl.metadata (6.2 kB)
Collecting annotated-types>=0.6.0 (from pydantic>=2.9->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.33.1 (from pydantic>=2.9->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached pydantic_core-2.33.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting typing-inspection>=0.4.0 (from pydantic>=2.9->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached typing_inspection-0.4.0-py3-none-any.whl.metadata (2.6 kB)
Collecting anyio<5,>=3.6.2 (from starlette<0.47.0,>=0.40.0->fastapi>=0.115.0->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading anyio-4.9.0-py3-none-any.whl.metadata (4.7 kB)
Collecting exceptiongroup>=1.0.2 (from anyio<5,>=3.6.2->starlette<0.47.0,>=0.40.0->fastapi>=0.115.0->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached exceptiongroup-1.2.2-py3-none-any.whl.metadata (6.6 kB)
Collecting sniffio>=1.1 (from anyio<5,>=3.6.2->starlette<0.47.0,>=0.40.0->fastapi>=0.115.0->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
Collecting fastapi-cli>=0.0.5 (from fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading fastapi_cli-0.0.7-py3-none-any.whl.metadata (6.2 kB)
Collecting httpx>=0.23.0 (from fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Collecting python-multipart>=0.0.18 (from fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading python_multipart-0.0.20-py3-none-any.whl.metadata (1.8 kB)
Collecting email-validator>=2.0.0 (from fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading email_validator-2.2.0-py3-none-any.whl.metadata (25 kB)
Collecting uvicorn>=0.12.0 (from uvicorn[standard]>=0.12.0; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading uvicorn-0.34.2-py3-none-any.whl.metadata (6.5 kB)
Collecting dnspython>=2.0.0 (from email-validator>=2.0.0->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading dnspython-2.7.0-py3-none-any.whl.metadata (5.8 kB)
Collecting typer>=0.12.3 (from fastapi-cli>=0.0.5->fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading typer-0.15.3-py3-none-any.whl.metadata (15 kB)
Collecting rich-toolkit>=0.11.1 (from fastapi-cli>=0.0.5->fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading rich_toolkit-0.14.3-py3-none-any.whl.metadata (999 bytes)
Collecting httpcore==1.* (from httpx>=0.23.0->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Collecting h11>=0.16 (from httpcore==1.*->httpx>=0.23.0->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting hf-xet>=0.1.4 (from huggingface-hub[hf_xet]>=0.30.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading hf_xet-1.0.5-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (494 bytes)
Requirement already satisfied: MarkupSafe>=2.0 in /data/project/wuzhongqin/.conda/envs/vllm_dev/lib/python3.10/site-packages (from jinja2->outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled) (3.0.2)
Collecting attrs>=22.2.0 (from jsonschema->outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached attrs-25.3.0-py3-none-any.whl.metadata (10 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema->outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading jsonschema_specifications-2025.4.1-py3-none-any.whl.metadata (2.9 kB)
Collecting rpds-py>=0.7.1 (from jsonschema->outlines==0.1.11->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading rpds_py-0.24.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
Collecting distro<2,>=1.7.0 (from openai>=1.52.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)
Collecting jiter<1,>=0.4.0 (from openai>=1.52.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading jiter-0.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.2 kB)
Collecting click>=7.0 (from ray!=2.44.*,>=2.43.0->ray[cgraph]!=2.44.*,>=2.43.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
Collecting msgpack<2.0.0,>=1.0.0 (from ray!=2.44.*,>=2.43.0->ray[cgraph]!=2.44.*,>=2.43.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached msgpack-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.4 kB)
Collecting aiosignal (from ray!=2.44.*,>=2.43.0->ray[cgraph]!=2.44.*,>=2.43.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached aiosignal-1.3.2-py2.py3-none-any.whl.metadata (3.8 kB)
Collecting frozenlist (from ray!=2.44.*,>=2.43.0->ray[cgraph]!=2.44.*,>=2.43.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading frozenlist-1.6.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
Collecting cupy-cuda12x (from ray[cgraph]!=2.44.*,>=2.43.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading cupy_cuda12x-13.4.1-cp310-cp310-manylinux2014_x86_64.whl.metadata (2.6 kB)
Collecting rich>=13.7.1 (from rich-toolkit>=0.11.1->fastapi-cli>=0.0.5->fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading rich-14.0.0-py3-none-any.whl.metadata (18 kB)
Collecting markdown-it-py>=2.2.0 (from rich>=13.7.1->rich-toolkit>=0.11.1->fastapi-cli>=0.0.5->fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich>=13.7.1->rich-toolkit>=0.11.1->fastapi-cli>=0.0.5->fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading pygments-2.19.1-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=13.7.1->rich-toolkit>=0.11.1->fastapi-cli>=0.0.5->fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Collecting regex>=2022.1.18 (from tiktoken>=0.6.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached regex-2024.11.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)
Collecting safetensors>=0.4.3 (from transformers>=4.51.1->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
Collecting shellingham>=1.3.0 (from typer>=0.12.3->fastapi-cli>=0.0.5->fastapi-cli[standard]>=0.0.5; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
Collecting httptools>=0.6.3 (from uvicorn[standard]>=0.12.0; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached httptools-0.6.4-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.6 kB)
Collecting python-dotenv>=0.13 (from uvicorn[standard]>=0.12.0; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading python_dotenv-1.1.0-py3-none-any.whl.metadata (24 kB)
Collecting uvloop!=0.15.0,!=0.15.1,>=0.14.0 (from uvicorn[standard]>=0.12.0; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached uvloop-0.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
Collecting websockets>=10.4 (from uvicorn[standard]>=0.12.0; extra == "standard"->fastapi[standard]>=0.115.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading websockets-15.0.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting aiohappyeyeballs>=2.3.0 (from aiohttp->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)
Collecting async-timeout<6.0,>=4.0 (from aiohttp->vllm==0.1.dev6130+g506475d.precompiled)
  Using cached async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading multidict-6.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.3 kB)
Collecting propcache>=0.2.0 (from aiohttp->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading propcache-0.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (10 kB)
Collecting yarl<2.0,>=1.17.0 (from aiohttp->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading yarl-1.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (72 kB)
Collecting fastrlock>=0.5 (from cupy-cuda12x->ray[cgraph]!=2.44.*,>=2.43.0->vllm==0.1.dev6130+g506475d.precompiled)
  Downloading fastrlock-0.8.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_28_x86_64.whl.metadata (7.7 kB)
Downloading compressed_tensors-0.9.3-py3-none-any.whl (98 kB)
Downloading depyf-0.18.0-py3-none-any.whl (38 kB)
Downloading lark-1.2.2-py3-none-any.whl (111 kB)
Downloading numba-0.61.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 19.5 kB/s eta 0:00:00
Downloading outlines-0.1.11-py3-none-any.whl (87 kB)
Downloading outlines_core-0.1.26-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (343 kB)
Using cached torch-2.6.0-cp310-cp310-manylinux1_x86_64.whl (766.7 MB)
Using cached nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
Using cached nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
Using cached nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
Using cached nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
Using cached nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
Using cached nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
Using cached nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
Using cached nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
Using cached nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
Using cached nvidia_cusparselt_cu12-0.6.2-py3-none-manylinux2014_x86_64.whl (150.1 MB)
Using cached nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl (188.7 MB)
Using cached nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
Using cached nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (99 kB)
Using cached sympy-1.13.1-py3-none-any.whl (6.2 MB)
Downloading torchaudio-2.6.0-cp310-cp310-manylinux1_x86_64.whl (3.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.4/3.4 MB 15.3 kB/s eta 0:00:00
Downloading torchvision-0.21.0-cp310-cp310-manylinux1_x86_64.whl (7.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.2/7.2 MB 18.0 kB/s eta 0:00:00
Using cached triton-3.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (253.1 MB)
Downloading xformers-0.0.29.post2-cp310-cp310-manylinux_2_28_x86_64.whl (44.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.3/44.3 MB 240.4 kB/s eta 0:00:00
Downloading xgrammar-0.1.18-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 118.6 kB/s eta 0:00:00
Downloading llguidance-0.7.19-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (14.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.0/14.0 MB 15.5 kB/s eta 0:00:00
Downloading llvmlite-0.44.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (42.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 42.4/42.4 MB 79.7 kB/s eta 0:00:00
Downloading lm_format_enforcer-0.10.11-py3-none-any.whl (44 kB)
Downloading opentelemetry_api-1.26.0-py3-none-any.whl (61 kB)
Downloading importlib_metadata-8.0.0-py3-none-any.whl (24 kB)
Downloading opentelemetry_exporter_otlp-1.26.0-py3-none-any.whl (7.0 kB)
Downloading opentelemetry_exporter_otlp_proto_grpc-1.26.0-py3-none-any.whl (18 kB)
Downloading opentelemetry_exporter_otlp_proto_common-1.26.0-py3-none-any.whl (17 kB)
Downloading opentelemetry_exporter_otlp_proto_http-1.26.0-py3-none-any.whl (16 kB)
Downloading opentelemetry_proto-1.26.0-py3-none-any.whl (52 kB)
Downloading googleapis_common_protos-1.70.0-py3-none-any.whl (294 kB)
Using cached grpcio-1.71.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.9 MB)
Downloading opentelemetry_sdk-1.26.0-py3-none-any.whl (109 kB)
Downloading opentelemetry_semantic_conventions-0.47b0-py3-none-any.whl (138 kB)
Downloading opentelemetry_semantic_conventions_ai-0.4.3-py3-none-any.whl (5.4 kB)
Downloading protobuf-4.25.7-cp37-abi3-manylinux2014_x86_64.whl (294 kB)
Downloading Deprecated-1.2.18-py2.py3-none-any.whl (10.0 kB)
Using cached wrapt-1.17.2-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (82 kB)
Downloading fastapi-0.115.12-py3-none-any.whl (95 kB)
Using cached pydantic-2.11.3-py3-none-any.whl (443 kB)
Using cached pydantic_core-2.33.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
Downloading starlette-0.46.2-py3-none-any.whl (72 kB)
Downloading anyio-4.9.0-py3-none-any.whl (100 kB)
Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
Using cached exceptiongroup-1.2.2-py3-none-any.whl (16 kB)
Downloading email_validator-2.2.0-py3-none-any.whl (33 kB)
Downloading dnspython-2.7.0-py3-none-any.whl (313 kB)
Downloading fastapi_cli-0.0.7-py3-none-any.whl (10 kB)
Downloading gguf-0.16.2-py3-none-any.whl (92 kB)
Using cached sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
Downloading httpcore-1.0.9-py3-none-any.whl (78 kB)
Downloading h11-0.16.0-py3-none-any.whl (37 kB)
Using cached huggingface_hub-0.30.2-py3-none-any.whl (481 kB)
Using cached fsspec-2025.3.2-py3-none-any.whl (194 kB)
Downloading hf_xet-1.0.5-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (54.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.0/54.0 MB 140.8 kB/s eta 0:00:00
Downloading interegular-0.3.3-py37-none-any.whl (23 kB)
Downloading mistral_common-1.5.4-py3-none-any.whl (6.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.5/6.5 MB 17.1 kB/s eta 0:00:00
Downloading jsonschema-4.23.0-py3-none-any.whl (88 kB)
Using cached attrs-25.3.0-py3-none-any.whl (63 kB)
Downloading jsonschema_specifications-2025.4.1-py3-none-any.whl (18 kB)
Downloading openai-1.76.0-py3-none-any.whl (661 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 661.2/661.2 kB 16.8 kB/s eta 0:00:00
Downloading distro-1.9.0-py3-none-any.whl (20 kB)
Downloading jiter-0.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (352 kB)
Downloading opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (50.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50.0/50.0 MB 220.6 kB/s eta 0:00:00
Using cached packaging-25.0-py3-none-any.whl (66 kB)
Downloading prometheus_client-0.21.1-py3-none-any.whl (54 kB)
Downloading prometheus_fastapi_instrumentator-7.1.0-py3-none-any.whl (19 kB)
Downloading python_multipart-0.0.20-py3-none-any.whl (24 kB)
Downloading pyzmq-26.4.0-cp310-cp310-manylinux_2_28_x86_64.whl (862 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 862.5/862.5 kB 124.2 kB/s eta 0:00:00
Downloading ray-2.43.0-cp310-cp310-manylinux2014_x86_64.whl (67.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.6/67.6 MB 278.8 kB/s eta 0:00:00
Using cached msgpack-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (378 kB)
Using cached click-8.1.8-py3-none-any.whl (98 kB)
Downloading referencing-0.36.2-py3-none-any.whl (26 kB)
Downloading rich_toolkit-0.14.3-py3-none-any.whl (24 kB)
Downloading rich-14.0.0-py3-none-any.whl (243 kB)
Downloading pygments-2.19.1-py3-none-any.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 359.3 kB/s eta 0:00:00
Using cached markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Downloading rpds_py-0.24.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (389 kB)
Using cached sniffio-1.3.1-py3-none-any.whl (10 kB)
Downloading tiktoken-0.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 197.4 kB/s eta 0:00:00
Using cached regex-2024.11.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (781 kB)
Using cached tokenizers-0.21.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
Using cached tqdm-4.67.1-py3-none-any.whl (78 kB)
Using cached transformers-4.51.3-py3-none-any.whl (10.4 MB)
Using cached safetensors-0.5.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (471 kB)
Downloading typer-0.15.3-py3-none-any.whl (45 kB)
Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
Using cached typing_inspection-0.4.0-py3-none-any.whl (14 kB)
Downloading uvicorn-0.34.2-py3-none-any.whl (62 kB)
Downloading httptools-0.6.4-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (442 kB)
Downloading python_dotenv-1.1.0-py3-none-any.whl (20 kB)
Downloading uvloop-0.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 142.0 kB/s eta 0:00:00
Downloading watchfiles-1.0.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (454 kB)
Downloading websockets-15.0.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (181 kB)
Using cached zipp-3.21.0-py3-none-any.whl (9.6 kB)
Downloading aiohttp-3.11.18-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 13.9 kB/s eta 0:00:00
Using cached async_timeout-5.0.1-py3-none-any.whl (6.2 kB)
Downloading multidict-6.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (219 kB)
Downloading yarl-1.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (333 kB)
Using cached aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Using cached aiosignal-1.3.2-py2.py3-none-any.whl (7.6 kB)
Downloading frozenlist-1.6.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (287 kB)
Downloading propcache-0.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (206 kB)
Downloading airportsdata-20250224-py3-none-any.whl (913 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 913.7/913.7 kB 16.1 kB/s eta 0:00:00
Downloading astor-0.8.1-py2.py3-none-any.whl (27 kB)
Downloading blake3-1.0.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (376 kB)
Downloading cachetools-5.5.2-py3-none-any.whl (10 kB)
Downloading cloudpickle-3.1.1-py3-none-any.whl (20 kB)
Downloading cupy_cuda12x-13.4.1-cp310-cp310-manylinux2014_x86_64.whl (104.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.6/104.6 MB 290.9 kB/s eta 0:00:00
Downloading fastrlock-0.8.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_28_x86_64.whl (53 kB)
Downloading dill-0.4.0-py3-none-any.whl (119 kB)
Downloading diskcache-5.6.3-py3-none-any.whl (45 kB)
Using cached einops-0.8.1-py3-none-any.whl (64 kB)
Downloading msgspec-0.19.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (211 kB)
Using cached nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)
Using cached ninja-1.11.1.4-py3-none-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (422 kB)
Downloading partial_json_parser-0.2.1.1.post5-py3-none-any.whl (10 kB)
Using cached psutil-7.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (277 kB)
Downloading py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
Downloading pycountry-24.6.1-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 471.4 kB/s eta 0:00:00
Downloading python_json_logger-3.3.0-py3-none-any.whl (15 kB)
Downloading scipy-1.15.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.6/37.6 MB 209.2 kB/s eta 0:00:00
Building wheels for collected packages: vllm
  Building editable for vllm (pyproject.toml) ... done
  Created wheel for vllm: filename=vllm-0.1.dev6130+g506475d.precompiled-0.editable-cp310-cp310-linux_x86_64.whl size=13341 sha256=c4b967a9f5241adffe5e777f31c8969c682855e757f773758ec9d4151fc59946
  Stored in directory: /tmp/pip-ephem-wheel-cache-sd3rt6kz/wheels/ca/b0/64/92cff2a7cc2ec13e865a4ac9abb31f66e399e2bb627df21442
Successfully built vllm
Installing collected packages: triton, sentencepiece, py-cpuinfo, nvidia-cusparselt-cu12, fastrlock, blake3, zipp, wrapt, websockets, uvloop, typing-inspection, tqdm, sympy, sniffio, shellingham, scipy, safetensors, rpds-py, regex, pyzmq, python-multipart, python-json-logger, python-dotenv, pygments, pydantic-core, pycountry, psutil, protobuf, propcache, prometheus_client, partial-json-parser, packaging, opentelemetry-semantic-conventions-ai, opencv-python-headless, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, ninja, nest_asyncio, multidict, msgspec, msgpack, mdurl, llvmlite, llguidance, lark, jiter, interegular, httptools, hf-xet, h11, grpcio, fsspec, frozenlist, exceptiongroup, einops, dnspython, distro, diskcache, dill, cupy-cuda12x, cloudpickle, click, cachetools, attrs, async-timeout, astor, annotated-types, airportsdata, aiohappyeyeballs, yarl, uvicorn, tiktoken, referencing, pydantic, opentelemetry-proto, nvidia-cusparse-cu12, nvidia-cudnn-cu12, numba, markdown-it-py, importlib_metadata, huggingface-hub, httpcore, googleapis-common-protos, gguf, email-validator, depyf, deprecated, anyio, aiosignal, watchfiles, tokenizers, starlette, rich, opentelemetry-exporter-otlp-proto-common, opentelemetry-api, nvidia-cusolver-cu12, lm-format-enforcer, jsonschema-specifications, httpx, aiohttp, typer, transformers, torch, rich-toolkit, prometheus-fastapi-instrumentator, opentelemetry-semantic-conventions, openai, jsonschema, fastapi, xgrammar, xformers, torchvision, torchaudio, ray, outlines_core, opentelemetry-sdk, mistral_common, fastapi-cli, compressed-tensors, outlines, opentelemetry-exporter-otlp-proto-http, opentelemetry-exporter-otlp-proto-grpc, opentelemetry-exporter-otlp, vllm
  Attempting uninstall: triton
    Found existing installation: triton 3.1.0
    Uninstalling triton-3.1.0:
      Successfully uninstalled triton-3.1.0
  Attempting uninstall: sympy
    Found existing installation: sympy 1.13.3
    Uninstalling sympy-1.13.3:
      Successfully uninstalled sympy-1.13.3
  Attempting uninstall: torch
    Found existing installation: torch 2.5.1
    Uninstalling torch-2.5.1:
      Successfully uninstalled torch-2.5.1
  Attempting uninstall: torchvision
    Found existing installation: torchvision 0.20.1
    Uninstalling torchvision-0.20.1:
      Successfully uninstalled torchvision-0.20.1
  Attempting uninstall: torchaudio
    Found existing installation: torchaudio 2.5.1
    Uninstalling torchaudio-2.5.1:
      Successfully uninstalled torchaudio-2.5.1
Successfully installed aiohappyeyeballs-2.6.1 aiohttp-3.11.18 aiosignal-1.3.2 airportsdata-20250224 annotated-types-0.7.0 anyio-4.9.0 astor-0.8.1 async-timeout-5.0.1 attrs-25.3.0 blake3-1.0.4 cachetools-5.5.2 click-8.1.8 cloudpickle-3.1.1 compressed-tensors-0.9.3 cupy-cuda12x-13.4.1 deprecated-1.2.18 depyf-0.18.0 dill-0.4.0 diskcache-5.6.3 distro-1.9.0 dnspython-2.7.0 einops-0.8.1 email-validator-2.2.0 exceptiongroup-1.2.2 fastapi-0.115.12 fastapi-cli-0.0.7 fastrlock-0.8.3 frozenlist-1.6.0 fsspec-2025.3.2 gguf-0.16.2 googleapis-common-protos-1.70.0 grpcio-1.71.0 h11-0.16.0 hf-xet-1.0.5 httpcore-1.0.9 httptools-0.6.4 httpx-0.28.1 huggingface-hub-0.30.2 importlib_metadata-8.0.0 interegular-0.3.3 jiter-0.9.0 jsonschema-4.23.0 jsonschema-specifications-2025.4.1 lark-1.2.2 llguidance-0.7.19 llvmlite-0.44.0 lm-format-enforcer-0.10.11 markdown-it-py-3.0.0 mdurl-0.1.2 mistral_common-1.5.4 msgpack-1.1.0 msgspec-0.19.0 multidict-6.4.3 nest_asyncio-1.6.0 ninja-1.11.1.4 numba-0.61.2 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-cusparselt-cu12-0.6.2 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.4.127 openai-1.76.0 opencv-python-headless-4.11.0.86 opentelemetry-api-1.26.0 opentelemetry-exporter-otlp-1.26.0 opentelemetry-exporter-otlp-proto-common-1.26.0 opentelemetry-exporter-otlp-proto-grpc-1.26.0 opentelemetry-exporter-otlp-proto-http-1.26.0 opentelemetry-proto-1.26.0 opentelemetry-sdk-1.26.0 opentelemetry-semantic-conventions-0.47b0 opentelemetry-semantic-conventions-ai-0.4.3 outlines-0.1.11 outlines_core-0.1.26 packaging-25.0 partial-json-parser-0.2.1.1.post5 prometheus-fastapi-instrumentator-7.1.0 prometheus_client-0.21.1 propcache-0.3.1 protobuf-4.25.7 psutil-7.0.0 py-cpuinfo-9.0.0 pycountry-24.6.1 pydantic-2.11.3 pydantic-core-2.33.1 pygments-2.19.1 python-dotenv-1.1.0 python-json-logger-3.3.0 python-multipart-0.0.20 pyzmq-26.4.0 ray-2.43.0 referencing-0.36.2 regex-2024.11.6 rich-14.0.0 rich-toolkit-0.14.3 rpds-py-0.24.0 safetensors-0.5.3 scipy-1.15.2 sentencepiece-0.2.0 shellingham-1.5.4 sniffio-1.3.1 starlette-0.46.2 sympy-1.13.1 tiktoken-0.9.0 tokenizers-0.21.1 torch-2.6.0 torchaudio-2.6.0 torchvision-0.21.0 tqdm-4.67.1 transformers-4.51.3 triton-3.2.0 typer-0.15.3 typing-inspection-0.4.0 uvicorn-0.34.2 uvloop-0.21.0 vllm-0.1.dev6130+g506475d.precompiled watchfiles-1.0.5 websockets-15.0.1 wrapt-1.17.2 xformers-0.0.29.post2 xgrammar-0.1.18 yarl-1.20.0 zipp-3.21.0
```

# 其他尝试
- 重新建立一个 conda环境，并且不安装 cuda / torch, 直接装 vllm
```bash
conda create -n vllm_dev_2 python=3.10
MAX_JOBS=16 VLLM_USE_PRECOMPILED=1 pip install -e .
# 
```
