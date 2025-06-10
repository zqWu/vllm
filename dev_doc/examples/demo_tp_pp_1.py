# 这个demo 展示 tp + pp

import torch

tensor_3_4 = torch.rand(3, 4)
tensor_4_2 = torch.rand(4, 2)


class LayerA(torch.nn.Module):
    def __init__(self):
        super(LayerA, self).__init__()
        self.weight = tensor_3_4

    def forward(self, x):
        assert x.shape[-1] == 3
        return x @ self.weight


class LayerB(torch.nn.Module):
    def __init__(self):
        super(LayerB, self).__init__()
        self.weight = tensor_4_2

    def forward(self, x):
        assert x.shape[-1] == 4
        return x @ self.weight


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = LayerA()
        self.layer2 = LayerB()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def simple_tp_pp(input_x):
    """
    手动进行 tp 和 pp
    tp: 对模型中的 tensor 进行拆分计算
    pp: 对模型中的 layer 进行拆分计算
    """
    model = MyNet()  # 假设已经知道 model的内部结构

    # 先对 layer1 进行 tp 计算
    layer1 = model.layer1
    layer1_weight = layer1.weight  # (3,4)
    weight_chunks = torch.split(layer1_weight, [3, 1], dim=-1)  # 拆成2块
    weight_chunk_1 = weight_chunks[0]  # (3,3)
    weight_chunk_2 = weight_chunks[1]  # (3,1)
    # 分别进行计算, 分别在不同的设备上
    # 模拟两个设备上进行 tp 计算
    layer1_out1 = input_x @ weight_chunk_1
    layer1_out2 = input_x @ weight_chunk_2
    # device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device0)
    # layer1_out1 = (input_x.to(device0) @ weight_chunk_1.to(device0)).to("cpu")
    # layer1_out2 = (input_x.to(device1) @ weight_chunk_2.to(device1)).to("cpu")
    layer1_out = torch.cat((layer1_out1, layer1_out2), dim=-1)  # tp 结束

    # 再对 layer2 进行 计算(略)
    layer2_out = layer1_out @ model.layer2.weight
    print(layer2_out)


def whole_model(input_x):
    model = MyNet()
    output = model(input_x)
    print(output)


if __name__ == '__main__':
    input_x = torch.randn(1, 3)

    whole_model(input_x)
    simple_tp_pp(input_x)
