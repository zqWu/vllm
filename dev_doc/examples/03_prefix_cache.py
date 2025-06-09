import argparse
import os

from dev_doc.examples.utils import Utils
from vllm import EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine.llm_engine import LLMEngine

gpu_id, _, free_mem_percent = Utils.get_single_gpu()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
os.environ["VLLM_USE_V1"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # A800
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # debug, 单进程更好


def parse_args(mock_cli_str):
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args(mock_cli_str)


def main(args: argparse.Namespace):
    """
    模拟 running: [1个 req], waiting = [1个 req]
    观察调度 + 执行
    """
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        n=1,
        temperature=1,
        top_p=0.95,
        max_tokens=999
    )
    # 超过 block-size=16, 观察 prefix match. 参考 test_prefix_caching.py
    # 前缀匹配 基于 block级别的 hash, 而不是单纯的 str匹配, 因此至少要1个block(16token)相同
    block_size = 16
    seqA_tokens = "hello " * (2 * block_size)  # SeqA
    seqB_tokens = seqA_tokens

    engine.add_request("req_id_1", seqA_tokens, sampling_params)
    engine.add_request("req_id_2", seqB_tokens, sampling_params)
    print(f"[debug] ================ curr_step_num:1 ============= ")
    os.environ["curr_step_num"] = "1"
    request_outputs = engine.step()
    # import os
    # if os.getenv("curr_step_num") == "1":
    #     print(f"[debug] {self.__class__.__name__}.xxx")

    print(f"[debug] ================ curr_step_num:2 ============= ")
    os.environ["curr_step_num"] = "2"
    request_outputs = engine.step()


if __name__ == '__main__':
    desc = """
观察step过程中的 相同的prefix 缓存复用
"""
    Utils.print_pid_tid()  # PYDEVD_USE_FRAME_EVAL
    mock_cli_str = [

        f"--enforce-eager",  # 这个模式能debug进源码, 否则 @support_torch_compile 断点进不了
        f"--model={Utils.get_model_path()}",
        # f"--gpu_memory_utilization={free_mem_percent - 0.05}",
        f"--gpu_memory_utilization=0.9",
        f"--swap-space=0",  # 禁止 swap到内存上
        # 使用一些特殊数字, 容易观察
        f"--max-model-len=44",
        f"--block-size=16",  # 16的倍数
        f"--max-num-seqs=5",
    ]
    args = parse_args(mock_cli_str)
    main(args)
