# offline batched inference

import os

from dev_doc.examples.utils import Utils
from vllm import LLM, SamplingParams

gpu_id, _, free_mem_percent = Utils.get_single_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
os.environ["VLLM_USE_V1"] = "0"

prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

# ===========================================================================
# 采样参数
# ===========================================================================
sampling_params = SamplingParams(
    n=1,
    temperature=0.8,
    top_p=0.95,
    max_tokens=2
)
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
llm = LLM(model=Utils.get_model_path(),
          gpu_memory_utilization=free_mem_percent - 0.05,
          swap_space=0.,  # 禁止swap到内存
          )

# ===========================================================================
# 推理
# ===========================================================================
outputs = llm.generate(prompts, sampling_params)

# ===========================================================================
# 对每一条prompt，打印其推理结果
# ===========================================================================
for output in outputs:
    prompt = output.prompt
    for ele in output.outputs:  # 一个prompt可能进行了多个 sampling
        generated_text = ele.text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
