# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser

if __name__ == "__main__":
    mock_cli_str = [
        "--enforce-eager",  # 这个模式能debug进源码, 否则 @support_torch_compile 断点进不了
        "--model=/data/models/Qwen2.5-0.5B-Instruct",
        # f"--model={Utils.get_model_path()}",
        # f"--gpu_memory_utilization={free_mem_percent - 0.05}",
        "--gpu_memory_utilization=0.99",
        "--swap-space=0",  # 禁止 swap到内存上
        # 使用一些特殊数字, 容易观察
        "--max-model-len=1000",
        "--block-size=16",
        "--max-num-seqs=5",
        "--port=9071",
        "--enable-auto-tool-choice",
        "--tool-call-parser=hermes",
    ]

    cli_env_setup()
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(mock_cli_str)
    validate_parsed_serve_args(args)

    import asyncio
    asyncio.run(run_server(args))
    # uvloop.run(run_server(args)) # uvloop debug会出错, 换成 asyncio
