
# refer
- 原理 https://lmsys.org/blog/2024-02-05-compressed-fsm/
- vllm+StructuredOutput https://www.youtube.com/watch?v=CGEkEwg0V9U  17:48 ~ 44:11
- ppt https://docs.google.com/presentation/d/1a5dHf3iRXSgbeOCa_TBaWujxq9B5EwEw
- 如何使用 https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html
- xgrammar https://github.com/mlc-ai/xgrammar



# 是什么
- 约束 output, 使之符合 "指定"的格式
- 可以通过 python api / OpenAI-compatible API Server


# pytorch
torch.Tensor.masked_fill_(mask, value)
掩码操作, 用value填充mask中值=1对应的元素
```python
import torch
t = torch.tensor([[1,2],[3,4]])
mask = torch.tensor([[True,False], [False,True]])
m = t.masked_fill(mask, 0)
# m = tensor([[0, 2], [3, 0]])
```

