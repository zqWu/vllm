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
- 验证ok
- 这个非常好, 能够适配多个包
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
