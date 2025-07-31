# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time

from transformers import AutoTokenizer

from dev_doc.examples.utils import Utils
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

gpu_id, _, free_mem_percent = Utils.get_single_gpu()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
os.environ["VLLM_USE_V1"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # A800

MODEL = Utils.get_model_path()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
SAMPLING_PARAMS = SamplingParams(
    n=1,
    temperature=1,
    top_p=0.95,
    max_tokens=999
)


def make_request(req_id: str, prompt: str) -> EngineCoreRequest:
    prompt_token_ids = TOKENIZER(prompt).input_ids
    return EngineCoreRequest(
        request_id=req_id,
        prompt_token_ids=prompt_token_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SAMPLING_PARAMS,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def main():
    engine_args = EngineArgs(
        model=Utils.get_model_path(),
        enable_prefix_caching=False,
        max_num_batched_tokens=10,
        # gpu_memory_utilization=0.9,
        gpu_memory_utilization=free_mem_percent - 0.1,
        swap_space=0,
        max_model_len=33,
        block_size=16,
        max_num_seqs=5,
    )
    vllm_config = engine_args.create_engine_config()
    engine_core = EngineCore(
        vllm_config=vllm_config,
        log_stats=True,
        executor_class=MultiprocExecutor,  # 指定多进程 Executor
    )

    prompt1 = "Write an engaging science fiction story about robots living alongside humans on Earth, exploring their conflicts."  # noqa
    prompt2 = "A dog chases after a rabbit"  # noqa
    req_1 = make_request("req_id_1", prompt1)
    req_2 = make_request("req_id_2", prompt2)
    engine_core.add_request(req_1)
    engine_core.add_request(req_2)

    print("[debug] ================ curr_step_num:1 ============= ")
    os.environ["curr_step_num"] = "1"
    request_outputs = engine_core.step()

    print("[debug] ================ curr_step_num:2 ============= ")
    os.environ["curr_step_num"] = "2"
    request_outputs = engine_core.step()


if __name__ == '__main__':
    main()
