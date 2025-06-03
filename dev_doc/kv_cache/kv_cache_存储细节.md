# 先讲整体

1. 每次 step()时
   计算tokens(可能是 prompt tokens, 也可能是 上次生成的token) 对应的 slot_mapping
   比如 slot_mapping = [32, 33, ...] 或 slot_mapping = [48,(这种间断的是不同batch的token) 64]

2. 在 flash_attn.forward()时
    - 上下文提供的信息(以key为例, value是完全一致)
        - tokens 对应的 key = Tensor(如 2, 12, 64) = 2个key, 12头, 64维
        - 对应的 slot_mapping(这个在 schedule时进行了计算)
            - slot_mapping 能够计算出对应的 tensor的位置
            - 如 slot_mapping = [48, 65]
            - 48 = k_cache[block=3, slot=0]  block_size=16
            - 65 = k_cache[block=4, slot=1]  block_size=16

    - 操作: 把这些 tensor放到 对应的位置
        - k_cache[block数][slot数] = key_tensor

# 更细节

- 测试代码: dev_doc/examples/03_schedule.py
- 在vllm/v1/attention/backends/flash_attn.py::line 581 可以看到细节
- 相关日志
```
INFO 06-03 15:51:13 [gpu_model_runner.py:1073] [debug] num_input_tokens=27
[debug] OPTForCausalLM.forward(4参数)
[debug] OPTModel.forward(4参数)
[debug] OPTDecoder.forward(4参数)
[debug] OPTDecoderLayer.forward(hidden_state)
[debug] OPTAttention.forward(hidden_state)
[debug] Attention.forward(q,k,v)
[debug] /data/project/wuzhongqin/github/vllm/vllm/attention/layer.py.unified_attention_with_output curr_step_num=1
[debug] 实例kv_cache值: key | block[2] | slot[1] | head[0] | [:3] = tensor([0., 0., 0.], device='cuda:0', dtype=torch.float16)
[debug] q/k/v.shape=torch.Size([27, 12, 64])=batch * n_head * head_dim
INFO 06-03 15:51:13 [flash_attn.py:593] [debug] FlashAttentionImpl.forward
[debug] curr_step_num=1. 存储之前 key_cache[3,2,0,:3]=tensor([0., 0., 0.], device='cuda:0', dtype=torch.float16)
[debug] curr_step_num=1. 存储之后 key_cache[3,2,0,:3]=tensor([ 1.2920, -0.9404,  2.4551], device='cuda:0', dtype=torch.float16)
```

- 相关解释
```txt
block=0是特殊 block, block=1 我手动屏蔽了(见 block_pool.py)
在 debug中, req_id_1 使用 blocks=[2,3], 有19个token
第1次 forward(prefill)前, 没有 cache, 因此 req_id_1: bloc[3]_slot[2] = 空
第1次 forward(prefill)后, 这些内容塞入 kv_cache, 因此 req_id_1: block[3]_slot[2] = 有值, 对应19th token的key
```
