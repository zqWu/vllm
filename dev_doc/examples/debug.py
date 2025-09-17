import sys
from vllm.entrypoints.cli.main import main

if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())

# 以下这些参数, 配置到运行命令行中
_ = """
serve 
--model=/data/models/opt-125m 
--gpu_memory_utilization=0.1 
--dtype=float16 
--max-model-len=2048 
--tensor-parallel-size=1
--host=0.0.0.0 
--port=9031
--enforce-eager 
"""
# 配置gpu CUDA_VISIBLE_DEVICES=5
# 启动成功后, http://host:port/docs
