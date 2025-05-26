import argparse
import os

from dev_doc.examples.utils import Utils
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine.llm_engine import LLMEngine

gpu_id, _, free_mem_percent = Utils.get_single_gpu()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
os.environ["VLLM_USE_V1"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # A800
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # debug, 单进程更好

if __name__ == '__main__':
    Utils.print_pid_tid()
    mock_cli_str = [
        f"--model={Utils.get_model_path()}",
        f"--gpu_memory_utilization={free_mem_percent - 0.05}",  # 默认0.9
        f"--swap-space=0",  # 禁止 swap到内存上
    ]

    from vllm import EngineArgs  # 在 环境变量后

    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args(mock_cli_str)

    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
