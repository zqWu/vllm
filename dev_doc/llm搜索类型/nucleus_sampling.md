# 也就是 top_p
- 为什么叫 nucleus sampling
  - the smallest set of tokens whose cumulative probability mass exceeds a threshold p
  - 这个部分是 *最核心最有代表性的部分* 就像原子核 nucleus 之于原子一样
  - 整个 token 空间 ~ 原子
  - top_p部分 token ~ 原子核

| 方法                   | 确定性  | 多样性  | 风险      | 可调节性 | 典型用途       |
| -------------------- | ---- | ---- | ------- | ---- | ---------- |
| Greedy Search        | ✅ 高  | ❌ 低  | 重复、死循环  | ❌ 无  | 结构性预测任务    |
| Top-k Sampling       | ⚖️ 中 | ⚖️ 中 | k 固定不灵活 | ✅ 有  | 简单文本生成     |
| **Nucleus Sampling** | ⚖️ 中 | ✅ 高  | 受 p 控制  | ✅ 有  | 对话、创意写作、代码 |
