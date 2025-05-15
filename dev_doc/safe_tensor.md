# safe tensor 特点
- safe：避免执行 python代码(不像 pickle), 防代码注入
- zero-copy: 支持内存映射 mmap, 无需将整个文件读入内存后再解压
- 高性能: 结构紧凑, 序列化速度快, 支持并发加载多个张量
- 跨平台、跨框架: pytorch / tensorflow / ggml 都支持

# 文件结构
- 文件名: model.safetensors
- 文件组织: https://huggingface.co/docs/safetensors/en/index

- .safetensors 由3部分 组成
	- 8 bytes: u64 int, 表示 header部分的长度
	- header: json utf8 string, 结构是 名称:信息
```json
{
	"tensor_name_1": {
		"dtype": "数据类型 如 F64 F32 F16 BF16 I64 I32 I16 I8 U8 BOOL",
		"shape": "list<int> 如 [1, 16,256]",
		"offsets": "[begin, end], 表示在data区的偏移量"
	},
	"model.layers.0.weight": {

	}
}
```
	- data
		这里的数据是二进制, offset [begin, end] 对应了 tensor_name_x 中的数据内容


# 代码示例
```python
# pip install safetensors
################################ 保存 tensor
import torch
from safetensors.torch import save_file

tensors = {
    "embedding": torch.zeros((2, 2)),
    "attention": torch.zeros((2, 3))
}
save_file(tensors, "model.safetensors")


################################ 加载全部 tensor
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

################################ 加载部分 tensor(多GPU并行每个只需要部分tensor)
from safetensors import safe_open

tensors = {}
with safe_open("model.safetensors", framework="pt", device=0) as f:
    tensor_slice = f.get_slice("embedding")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]