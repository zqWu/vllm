import argparse
import os

from dev_doc.examples.utils_gpu import get_single_gpu
from vllm import EngineArgs, LLMEngine
from vllm.utils import FlexibleArgumentParser

gpu_id, gpu_free_mem, free_mem_percent = get_single_gpu()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
os.environ["VLLM_USE_V1"] = "0"


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_args(mock_cli_str):
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)

    if mock_cli_str:
        return parser.parse_args(mock_cli_str)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)


if __name__ == '__main__':
    mock_cli_str = [
        "--model=/data/models/opt-125m",
        "--gpu_memory_utilization=0.1",
    ]
    args = parse_args(mock_cli_str)
    main(args)
