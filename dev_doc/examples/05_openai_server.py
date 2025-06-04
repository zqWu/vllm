from dev_doc.examples.utils import Utils
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import (make_arg_parser, validate_parsed_serve_args)
import uvloop

if __name__ == "__main__":
    mock_cli_str = [
        f"--enforce-eager",  # 这个模式能debug进源码, 否则 @support_torch_compile 断点进不了
        f"--model=/data/models/Qwen2.5-0.5B-Instruct",
        # f"--model={Utils.get_model_path()}",
        # f"--gpu_memory_utilization={free_mem_percent - 0.05}",
        f"--gpu_memory_utilization=0.99",
        f"--swap-space=0",  # 禁止 swap到内存上
        # 使用一些特殊数字, 容易观察
        f"--max-model-len=1000",
        f"--block-size=16",
        f"--max-num-seqs=5",
        f"--port=9071",
        f"--enable-auto-tool-choice",
        f"--tool-call-parser=hermes",
    ]

    cli_env_setup()
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(mock_cli_str)
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
    "python vllm/entrypoints/openai/api_server.py --model [模型路径] [其他参数]"
