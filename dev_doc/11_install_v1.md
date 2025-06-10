```bash
#conda env remove -n vllm_dev -a
conda create -n vllm_dev python=3.10 -y
conda activate vllm_dev

cd github/vllm
# 分支是 main 分支, 大约是 v0.9.0

# 未在此环境中安装其他 pip / conda 包
export MAX_JOBS=16
VLLM_USE_PRECOMPILED=1 pip install --editable .
# 安装了 nvidia-xxx-cu12 等包
```

检查

```
>>> import torch

>>> torch.__version__
'2.7.0+cu126'

>>> torch.cuda.is_available()
True

>>> torch.version.cuda
'12.6'
```

# 指定 torch版本 + 安装 vllm

- 假设已经安装了 torch torchvision torchaudio等

```
git clone https://github.com/vllm-project/vllm.git
cd vllm

python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```

# 安装 lm-cache + vllm

## 安装方式 1 - fail

```
conda create -n vllm_dev python=3.10 -y
conda activate vllm_dev

# torch==2.6.0, cuda=12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install lmcache==0.2.1

# 在安装 vllm
python use_existing_torch.py
pip install -r requirements/build.txt
export MAX_JOBS=16
pip install --no-build-isolation -e .
```

## 安装方式 1 - fail - 细节

- 安装后 vllm 能正常工作, 但是 lmcache报错

```text
    import lmcache.c_ops as lmc_ops
ImportError: <xxx>/.conda/envs/vllm_dev/lib/python3.10/site-packages/lmcache/c_ops.cpython-310-x86_64-linux-gnu.so: 
undefined symbol: _ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_RKSs
```

- 这种错误原因是: lmcache 的 C++ 扩展编译时用的 libtorch 与你当前 Python 环境中的 torch 版本不兼容。
- 这个符号 _ZN3c106detail23torchInternalAssertFail... 是 torch 内部的 C++ 符号，它在不同版本中可能会变。

## 安装方式 2 - fail - 使用lmcache-v0.2.1源码进行安装, 错误与方式1一致

- 完全干净环境, 优先安装 lmcache
- 难绷, 此时torch/cuda版本都是 lmcache自己定的

## 安装方式 3

- 修改 python=3.12

```
conda create -n vllm_dev python=3.12 -y
conda activate vllm_dev

# step: 安装 lmcache
git clone https://github.com/LMCache/LMCache.git
cd LMCache
git checkout v0.2.1
uv pip install -e .
# 查看torch+cuda版本
# python -c "import torch; print(torch.__version__)" 
# 2.6.0+cu124


# step: 测试1
python -c "from lmcache.experimental.cache_engine import LMCacheEngineBuilder"
# 根据错误补安装, 直到测试1通过 
pip install py-cpuinfo


# step: 源码安装 vllm
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
export MAX_JOBS=16
pip install --no-build-isolation -e .

# step: 测试 lmcache + vllm, 不报错就通过
python3 -c "import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"
```