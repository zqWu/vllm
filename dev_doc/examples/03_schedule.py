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
        max_tokens=2
    )
    prompt1 = "write a scientific fiction"
    engine.add_request("req_id_1", prompt1, sampling_params)
    request_outputs = engine.step()

    engine.add_request("req_id_2", "Dogs love", sampling_params)
    request_outputs = engine.step()


if __name__ == '__main__':
    Utils.print_pid_tid()  # PYDEVD_USE_FRAME_EVAL
    mock_cli_str = [
        f"--model={Utils.get_model_path()}",
        # f"--gpu_memory_utilization={free_mem_percent - 0.05}",
        f"--gpu_memory_utilization=0.9",
        f"--swap-space=0",  # 禁止 swap到内存上
    ]
    args = parse_args(mock_cli_str)
    main(args)
