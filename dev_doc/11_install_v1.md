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


# 指定 torch版本等
- 假设已经安装了 torch torchvison torchaudio等
```
git clone https://github.com/vllm-project/vllm.git
cd vllm

python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```
