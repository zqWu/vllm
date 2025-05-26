from transformers import AutoModelForCausalLM
from torchinfo import summary
import torch

# 加载 OPT-125M 模型
model = AutoModelForCausalLM.from_pretrained("/data/models/opt-125m")

# 设置模型到 CPU 或 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# 使用 torchinfo 打印结构和 shape
# OPT 输入是 (batch, sequence_length)，embedding size 是 768
summary(model, input_size=(1, 32), dtypes=[torch.long])
