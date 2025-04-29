# offline batched inference

import os
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

# ===========================================================================
# 采样参数
# ===========================================================================
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# top_p 控制从语言模型的输出概率分布中选择哪些 token 用于下一步生成。
# 你: 0.3
# 我: 0.2
# the: 0.15
# is: 0.1
# ...
# top_p=1.0：保留全部 token，相当于不裁剪（=普通采样）。
# top_p=0.9：从最大概率开始向下加，直到累计概率 ≥ 0.9 为止


# ===========================================================================
# 初始化vLLM offline batched inference实例，并加载指定模型
# ===========================================================================
# "/data/models/opt-125m" "facebook/opt-125m"
llm = LLM(model="/data/models/opt-125m")

# ===========================================================================
# 推理
# ===========================================================================
outputs = llm.generate(prompts, sampling_params)

# ===========================================================================
# 对每一条prompt，打印其推理结果
# ===========================================================================
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
