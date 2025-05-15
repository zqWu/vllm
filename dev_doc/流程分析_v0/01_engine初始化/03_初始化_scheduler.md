=============
vllm_config 包含了所有配置信息

engine:LLMEngine
    |__ model_executor:UniProcExecutor或其他
    |   |__ driver_worker:WorkerWrapperBase
    |       |__ worker:vllm.worker.worker.Worker
    |           |__ model_runner: GPUModelRunnerBase -> ModelRunner
    |               |__ model: nnModule 具体的LLM类, 比如 OPTForCausalLM
    |
    |__ scheduler: List[Scheduler]
        |__ waiting:deque
        |__ running:deque
        |__ swapped:deque

=============
@dataclass
class SchedulerOutputState:
    """Caches the scheduler outputs for a virtual engine. Used for Multi-Step"""
    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None
    allow_async_output_proc: bool = False
    last_output: Optional[SamplerOutput] = None

=============
初始化  model_executor之后, 初始化 scheduler
```python
# LLMEngine.__init__

# 初始化 model_executor
# Scheduler = 确定 Scheduler 所使用的 class
self.scheduler = [ Scheduler(...) for v_id in range(pipeline_parallel_数量)]

```

=============
```
Scheduler.__init__
    self.block_manager = ...
    self.waiting: Deque[SequenceGroup] = deque()
    self.running: Deque[SequenceGroup] = deque()
    self.swapped: Deque[SequenceGroup] = deque()
    self._finished_requests_ids: List[str] = list()
    self.cache_id = 0

方法
free_seq(self, seq)
schedule(slef)
free_finished_seq
free_finished_seq_group

fork_seq(self, parent_seq, child_seq)
``` 
