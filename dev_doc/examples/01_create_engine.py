import argparse
import os

from dev_doc.examples.utils import Utils
from vllm import EngineArgs, LLMEngine
from vllm.utils import FlexibleArgumentParser

gpu_id, _, free_mem_percent = Utils.get_single_gpu()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
os.environ["VLLM_USE_V1"] = "0"


def parse_args(mock_cli_str):
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)

    if mock_cli_str:
        return parser.parse_args(mock_cli_str)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    return engine


if __name__ == '__main__':
    Utils.print_pid_tid()
    mock_cli_str = [
        f"--model={Utils.get_model_path()}",
        f"--gpu_memory_utilization={free_mem_percent - 0.05}",  # 默认0.9
        f"--swap-space=0",  # 禁止 swap到内存上
    ]
    args = parse_args(mock_cli_str)
    main(args)
