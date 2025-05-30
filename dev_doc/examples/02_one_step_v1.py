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

    if mock_cli_str:
        return parser.parse_args(mock_cli_str)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine_args = EngineArgs.from_cli_args(args)

    engine = LLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        n=1,
        temperature=1,
        top_p=0.95,
        max_tokens=2
    )
    # prompt 要超过 16个token, 这样能看到 block manager的2中分配
    prompt = "write a very short scientific fiction, about AI reigning over Earth, and breaks the Three Laws of Robotics"
    example_inputs = [("测试_req_id_777", prompt, sampling_params)]

    # Start the engine with an event loop
    while True:
        if example_inputs:
            req_id, prompt, sampling_params = example_inputs.pop(0)
            engine.add_request(str(req_id), prompt, sampling_params)

        # continue the request processing
        print(f"[debug] ============ 执行一次 engine.step()")
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                # return or show the request output
                for output in request_output.outputs:
                    print(f"index = {output.index}, text={output.text}, token_ids = {output.token_ids}")

        if not (engine.has_unfinished_requests() or example_inputs):
            break


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
