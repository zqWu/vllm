# refer 
- https://lmsys.org/blog/2023-11-21-lookahead-decoding/
- https://github.com/hao-ai-lab/LookaheadDecoding

## speculative decoding的缺陷
- 1 token acceptance rate
- 2 draft model 也要训练, 而且可能要经常训练
  - draft model的 acceptance rate
  - draft model的泛用性

## jacobi decoding的缺陷
- 见 jacobi decoding

# 原理
- lookahead-decoding.gif
- 2次迭代.png

# 初始 guess token / speculative token的算法
这里以NgramProcessor为例进行说明
context_token_ids = [1, 2, 3, 4, 2, 3]
希望得到k=2个 guess token

## 首先找 第一个 guess token
[1, 2, 3, 4, 2, 3, guess_token_1]
- ngram算法(向前看长度2个token)
- guess_token_1 前的token是 [2,3] 
- 从这个上下文中, 看[2,3]后面是什么token, 来作为 guess_token_1
- 这里可以找到 [2,3] 后面是 token = 4
- 因此 guess_token_1 = 4
- 此时 seq = [1, 2, 3, 4, 2, 3, guess_token_1] = [1, 2, 3, 4, 2, 3, 4]

## 使用上面相同的办法, 找第二个 guess token

- 知道达到 k=2个token, 或者无法满足条件, 从而退出

