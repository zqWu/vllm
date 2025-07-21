# 目标: 掌握 vllm + distributed-executor-backend

- 有2个节点(172.31.0.3 和 172.31.0.2)，每个节点有8张A800的。这些ip都是绑定 eth0设备
- 使用 vllm在2台机器上部署 Qwen3-0.6B, 要求跨节点部署

# 背景说明

vLLM 支持通过 Ray 和 MPI 两种方式来实现多节点/多GPU分布式推理：
--distributed-executor-backend ray: 使用 Ray 来调度多个 GPU 和节点。
--distributed-executor-backend mpi: 使用 MPI (一般用于集群或 HPC 系统，需要 mpirun 或 torchrun 等工具)。

对于云服务器或常规多节点场景，推荐使用 Ray，配置相对简单、支持自动资源发现，并已是 vLLM 默认推荐方式。

## 详细步骤

### step: 所有节点安装 vLLM + Ray

```
pip install "vllm[serve]" ray
```

确保：
- Python 环境一致
- 所有机器之间能 互相 SSH 通信
- CUDA 驱动可用，能 nvidia-smi 正常显示

### step: 启动 ray
- 重要: ray进程启动时要指定 gloo使用的网卡, 不然可能 vllm报错 `RuntimeError: Gloo connectFullMesh failed`
- vllm + ray + gpu环境中, 有2种通讯
  - 初始时, torch.distributed.new_group(ranks, backend="gloo"), torch创建一个 gloo backend 来做 control plane
  - gpu之间通讯使用 nccl

```
export GLOO_SOCKET_IFNAME=eth0
ray start --head --node-ip-address=172.31.0.3 --port=6379
# ray status 检查状态

# 加入集群

export GLOO_SOCKET_IFNAME=eth0
ray start --address=172.31.0.3:6379 --node-ip-address=172.31.0.2
```

### 检查集群, 2个节点都正常

```bash
(wzq_vllm) wuzhongqin@:~/vllm$ ray status
======== Autoscaler status: 2025-06-23 12:14:02.169277 ========
Node status
---------------------------------------------------------------
Active:
 1 node_3e28f1812f44a5120832423a3a2dafeb323889124955b4be13243998
 1 node_046c570b53d768ee0582924bce9c860719f706786e99defbbbd80b24
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Total Usage:
 0.0/256.0 CPU
 0.0/16.0 GPU
 0B/3.48TiB memory
 0B/372.53GiB object_store_memory

Total Constraints:
 (no request_resources() constraints)
Total Demands:
 (no resource demands)
```

### step: vLLM 启动分布式推理
1. 在ray集群上的任意节点执行

```bash
# ##################### 成功
export NCCL_DEBUG=TRACE
export GLOO_LOG_LEVEL=TRACE
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

vllm serve /data/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 16 \
  --distributed-executor-backend ray \
  --port 9081

vllm serve /data/models/Qwen3-0.6B \
  --served-model-name Qwen3-0.6B \
  --tensor-parallel-size 8 \
  ----pipeline-parallel-size 2\
  --distributed-executor-backend ray \
  --port 9081
```
