import sys
from vllm.entrypoints.cli.main import main

if __name__ == '__main__':
    if sys.argv[0].endswith('.exe'):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())


_ = """
# 1. 以下这些参数, 配置到运行命令行中
serve 
--model=/data/models/opt-125m   --gpu_memory_utilization=0.1
--dtype=float16 
--max-model-len=2048 
--tensor-parallel-size=1
--host=0.0.0.0 
--port=9031
--enforce-eager


# 常用配置参数
--model=/data/models/opt-125m   --gpu_memory_utilization=0.1 
--model=/data/models/Qwen3-0.6B --gpu_memory_utilization=0.3 


# 2. 配置gpu, 要配置环境变量. CUDA_VISIBLE_DEVICES=5


# 3. 简单测试
curl http://localhost:9031/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "将一个简短的冷笑话。"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
"""
