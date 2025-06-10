import torch
import torch.nn as nn


# 支持多 GPU 的设备分配器
def get_devices(n):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = [torch.device(f"cuda:{i}") for i in range(min(n, num_gpus))]
    if len(devices) < n:
        devices.extend([torch.device("cpu")] * (n - len(devices)))
    return devices


# 层 A： 3 -> 4
class LayerA(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return x @ self.weight


# 层 B： 4 -> 2
class LayerB(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return x @ self.weight


# 模型 = LayerA + LayerB
class MyNet(nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.layer1 = LayerA(w1)
        self.layer2 = LayerB(w2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# 模拟张量并行
def tensor_parallel_layer1(input_x, weight, devices):
    # weight shape: (3, 4)
    # 拆成两份
    chunks = torch.chunk(weight, len(devices), dim=-1)  # [(3,2), (3,2)]

    outputs = []
    for i, chunk in enumerate(chunks):
        device = devices[i]
        x = input_x.to(device)
        w = chunk.to(device)
        y = (x @ w).to("cpu")  # 模拟 device 并行后回收
        outputs.append(y)

    # 聚合结果
    return torch.cat(outputs, dim=-1)


# 模拟流水线并行（交错执行）
def pipeline_parallel(model, batch1, devices):
    # PP stage 1：batch1 经过 layer1
    layer1_device = devices[0]
    layer_1 = model.layer1.to(layer1_device)
    layer_1_input = batch1.to(layer1_device)
    layer_1_output = layer_1(layer_1_input).to("cpu")

    # PP stage 2：batch1 进入 layer2
    layer2_device = devices[1]
    layer_2 = model.layer2.to(layer2_device)
    layer_2_input = layer_1_output.to(layer2_device)
    layer_1_output = layer_2(layer_2_input).to("cpu")

    return layer_1_output


if __name__ == "__main__":
    # 固定权重
    w1 = torch.rand(3, 4)
    w2 = torch.rand(4, 2)
    model = MyNet(w1, w2)

    # 生成两个 batch
    batch1 = torch.randn(1, 3)
    devices = get_devices(2)

    # 原模型 + 权重 (默认都在 CPU上)
    print("\n=== Whole Model ===")
    print("WholeModel Output:", model(batch1))

    # TP: 模拟 layer1 在多个设备上执行
    print("\n=== 模拟 TP for Layer1 ===")
    layer1_out_tp = tensor_parallel_layer1(batch1, w1, devices)
    layer2_out = layer1_out_tp @ w2
    print("TP+Layer2 Output:", layer2_out)

    # PP: 模拟两个 batch 在流水线中并行执行
    print("\n=== 模拟 PP for MyNet ===")
    out1 = pipeline_parallel(model, batch1, devices)
    print("Batch1 Output:", out1)
