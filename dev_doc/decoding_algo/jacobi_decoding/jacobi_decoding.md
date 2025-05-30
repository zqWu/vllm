# refer
https://github.com/teelinsan/parallel-decoding
https://arxiv.org/pdf/2305.10427

这里有一张动图，非常好
https://lmsys.org/blog/2023-11-21-lookahead-decoding/

# 一些概念
decoder_input_ids 和 input_ids


## 对于 decoder-only的架构, 只需要 input_ids
input_ids就是 生成序列 的上下文
也是 decoder的输入

## 对于 T5 / BART(encoder-decoder模型)
- input_ids 给 encoder的输入
- decoder_input_ids = 给decoder的输入

例子:
```python
input_ids = tokenizer("How are you?", return_tensors="pt").input_ids
decoder_input_ids = tokenizer("Comment", return_tensors="pt").input_ids  # 初始解码器输入
```
- input_ids 是模型需要处理的源内容（英文）
- decoder_input_ids 是模型当前已经生成的目标内容（法语），用于指导生成下一个 token

## 核心代码理解
```python
        for index in range(str_index, max_length):
            if self.use_cache and index > 0:
                ...
            else:
                # 把 old_init_tensor 保留下来用于后续比较生成结果
                old_init_tensor = init_tensor.detach().clone()
                output = self.model(
                    input_ids,          # 给 encoder的输入
                    attention_mask,     # input_ids 的 attention_mask
                    decoder_input_ids=init_tensor, # decoder的输入
                    use_cache=True,
                )

            # logits 是 decoder的输出, softmax + argmax后得到 token 的 index
            logits = output.logits
            max_index = torch.argmax(logits, dim=-1)
            max_value, max_i = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            
            # 
            if index > 0 and logits_preprocessor is not None:
                logits_new = logits_preprocessor(total_res[:, : index + 1], logits[:, 0, :])
                max_value_new = torch.argmax(logits_new, dim=-1)
                max_index[:, 0] = max_value_new
            if self.use_cache and index > 0:
                init_tensor = max_index
                total_res = torch.cat((total_res[:, : index + 1], init_tensor[:, :-1]), dim=1)
            else:
                # 更新 init_tensor
                # 每次迭代后, init_tensor都会 length减一
                init_tensor[:, index + 1 :] = max_index[:, index:-1] 
                # total_res 用于表示比对的 tensor
                total_res = init_tensor
                output_probs[:, index + 1 :] = max_value[:, index:-1]

            stop_condition, return_tensor = self.stopping_criterion(old_init_tensor, total_res)
        return return_tensor, index


# 比对 past_tensor, current_tensor, 在 eos前都相同即可
def stopping_criterion(past_tensor, current_tensor, eos=None):
    assert past_tensor.shape == current_tensor.shape
    if torch.equal(past_tensor, current_tensor):
        tensor = current_tensor
        if eos is not None:
            if eos in current_tensor[0]:
                pos = (current_tensor[0] == eos).nonzero(as_tuple=True)[0]
                if pos.shape[0] > 1:
                    pos = pos[0].item()
                else:
                    pos = pos.item()
                return True, tensor, pos
            else:
                return True, tensor, -1
        return True, tensor
    else:
        if eos is not None:
            return False, current_tensor, False
        else:
            return False, current_tensor
```