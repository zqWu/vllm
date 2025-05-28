from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import torch

# 加载 OPT-125M 模型
model = AutoModelForCausalLM.from_pretrained("/data/models/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("/data/models/opt-125m")
device = "cpu"


def print_shape_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, torch.Tensor):
            print(f"{name:<60} -> {output.shape}")
        else:
            # final output
            last_hidden_state = output.last_hidden_state
            print(f"{name:<60} -> {last_hidden_state.shape}")

    return hook


# 注册 hook
for name, module in model.named_modules():
    if name and not isinstance(module, torch.nn.ModuleList):
        module.register_forward_hook(print_shape_hook(name))

# 准备输入

inputs = tokenizer("dogs like", return_tensors="pt").to(device)

# Forward
with torch.no_grad():
    _ = model(**inputs)
