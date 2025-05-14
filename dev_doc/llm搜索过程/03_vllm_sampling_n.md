# vllm 中 samplingConfig(n=2)
- 不是 beam search, 除非显示要求 (n=2, use_beam_search=True)

## 举例说明过程
```
prompt = "我喜欢"
SamplingConfig(
    n=2,
    temperature=1.0,
    top_p=0.9,
    use_beam_search=False  # 默认就是 False
)
```

## Sample1 - token1
Token   概率（未经温度和top-p处理）
吃       0.40
看       0.25
玩       0.20
读       0.10
写       0.05

模型采样选中了 "玩"，然后继续生成：
top_p=0.9 会截断到前几个累积概率达到 0.9 的 token：
可采样的 token 是：[吃, 看, 玩, 读]（这几个的累计概率是 0.40 + 0.25 + 0.20 + 0.10 = 0.95）

假设随机采样选中 "玩"

### Sample1 - token2
模型以 "我喜欢玩" 为上下文，预测下一个 token。

假设模型输出的 logits 映射为如下概率（未归一化）：
token   原始概率（softmax 后）
游戏          0.35
手机          0.25
电脑          0.20
滑板          0.10
沙子          0.05
土豆          0.05

使用 top_p=0.9 得到 候选集 = ["游戏", "手机", "电脑", "滑板"]
假设采样随机性作用下，选中了：“游戏”


## 第二次采样（Sample 2）
完全独立于第一次采样，过程与 sample 1 一样


# 实测1
```python
# 模型 facebook/opt-125m
sampling_params = SamplingParams(
    n=4,
    temperature=0.1,
    top_p=0.5,
    max_tokens=2
)
prompt = "I like playing"
```

index = 0, text= with my, token_ids = (19, 127)
index = 1, text= with my, token_ids = (19, 127)
index = 2, text= with my, token_ids = (19, 127)
index = 3, text= with my, token_ids = (19, 127)

# 实测2
```python
# 模型 facebook/opt-125m
sampling_params = SamplingParams(
    n=4,
    temperature=1, # 这里修改了
    top_p=0.5,
    max_tokens=2
)
prompt = "I like playing"
```

index = 0, text= with my, token_ids = (19, 127)
index = 1, text= a ", token_ids = (10, 22)
index = 2, text= PUB, token_ids = (221, 12027)
index = 3, text= with friends, token_ids = (19, 964)
