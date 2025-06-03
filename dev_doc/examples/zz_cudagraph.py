import torch
import torch.nn as nn
import os

#
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 设定设备
device = torch.device("cuda:5")

# 构造一个简单的模型
model = nn.Sequential(
    nn.Linear(3, 3),
    nn.ReLU(),
    nn.Linear(3, 3)
).to(device)

# 固定输入尺寸
input_tensor = torch.randn(3, 3, device=device)
target = torch.randn(3, 3, device=device)

# 损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 1. Warm-up 一次，记录静态参数
static_input = torch.randn_like(input_tensor)
static_target = torch.randn_like(target)

# 2. 使用 CUDA Graph 捕获前向 + 反向传播
g = torch.cuda.CUDAGraph()

# 构造 static 的 input/output 和 loss tensor
static_input = static_input.clone()
static_target = static_target.clone()

# 设置模型为 capture-ready 模式（使用 capture_stream）
model.train()
optimizer.zero_grad(set_to_none=True)

# warm-up: 触发 cuBLAS 初始化
with torch.no_grad():
    _ = model(static_input)  # 任何用到 Linear 层的前向推理都可以

# 开始捕获图
with torch.cuda.graph(g):
    output = model(static_input)
    loss = loss_fn(output, static_target)
    loss.backward()
    optimizer.step()

# 3. 之后的每次训练直接 replay CUDA Graph（速度更快）
for _ in range(10):
    static_input.copy_(input_tensor)
    static_target.copy_(target)
    g.replay()
