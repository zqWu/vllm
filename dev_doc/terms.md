#
- temperature = 2: 对于一个非归一化概率,统一 除以温度后, 进行softmax [1.0/t, 3.3/t, 3e2/t, ...]
  - t < 1, 变尖锐, 保守+确定性
  - t > 1, 变平缓, 发散+创新性
- top_k = 5: 概率从大到小, 取前5
- top_p = 0.9: 概率从大到小取, 总和刚好跨过0.9
- log_probs = 3: 推理后，去前三token，计算 log probability

# tp pp

## tp = tensor parallel
比如一个 FFN, 5 * 2, input = 3 * 5
```
                  t1 t2
r1 r1 r1 r1 r1    t1 t2
r2 r2 r2 r2 r2    t1 t2
r3 r3 r3 r3 r3    t1 t2
                  t1 t2
```
可以进行这样处理
```
                  t1
r1 r1 r1 r1 r1    t1
r2 r2 r2 r2 r2    t1
r3 r3 r3 r3 r3    t1
                  t1
```

```
                  t2
r1 r1 r1 r1 r1    t2
r2 r2 r2 r2 r2    t2
r3 r3 r3 r3 r3    t2
                  t2
```

## pp = pipeline parallel
