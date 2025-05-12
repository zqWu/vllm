# pytorch 正确的测试时间代码
```python
torch.cuda.synchronize()
start = time.time()
result = model(input)
torch.cuda.synchronize()
end = time.time()
```

在 torch中, 程序的执行都是异步的
torch.cuda.synchronize() 等待异步执行完成
