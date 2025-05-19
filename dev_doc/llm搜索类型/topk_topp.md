# top_k top_p
- top_k = 3, 只在几率最高的3个 token中选择
- top_p = 0.8, token按照几率排布, 累计几率 >= 0.8 (nucleus sampling)

# 一起配置时, 怎么起作用
- 最多从 top_k中取样
- 这 top_k 几率相加 < top_p

最终限制是2个条件的交集

- top_k 很小, top_p 再大也没用
- top_p 很小, top_k 再大也没用