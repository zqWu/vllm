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
SequenceData
    _prompt_token_ids
    _output_token_ids
    ======以下是生成过程中处理的数据=====
    _cumulative_logprob
    _prompt_token_ids_tuple
    _num_computed_tokens        计算了多少token
    _num_cached_tokens          prefix cache hit
    _stage                      enum: prefill or decode
    _cached_all_token_ids
    _new_appended_tokens
    _mrope_position_delta       gpt4o: 当前位置编码相对于缓存开头的位置差异

Sequence
    seq_id
    inputs
    block_size
    eos_token_id
    status = 默认WAITING | RUNNING | SWAPPED | FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / FINISHED_ABORTED / FINISHED_IGNORED
    ============
    is_prefill()
    is_finished()

SequenceGroup
    seqs = List[Sequence]
    prompt
    promt_token_ids
    metrics = {arrival_time | last_token_time | first_token_time | first_scheduled_time | time_in_queue | ...}
    state = {num_steps=1, current_step=0}
    cached_request_output = None
    ================
    is_finish()
    is_prefill()
    num_finished_seq()

=============
BlockPool:
    _pool:[]
    _free_ids: Deque[int]
    free_block(block)
    init_block(...)

=============
NaiveBlock:
    _prev_block
    _token_ids:[]
    _block_id: physical block_id, 由 NaiveBlockAllocator中生成, 并分配给Block(分配方式:第一个free的)
    _pool_id: 该block在 pool中的id (或序号)
    ================
    num_empty_slot()
    is_full()
    append_token_ids(...)


=============
BlockTable:
    _blocks
    _allocator
    _block_size
    _num_full_slots
    ==================
    _get_all_token_ids()
    fork()
    free()


=============
llm.generate(prompt="Hello, my name is", sampling_params=取样参数)

    # 第一步: 封装 SequenceGroup, 添加到 waiting队列
    LLM._validate_and_add_requests(prompt, sampling_params, 其他参数略)
        LLM._add_request(prompt, sampling_params, 其他参数略)
            request_id = counter()
            self.llm_engine.add_request(request_id, prompt, params=取样参数)
            LLMEngine.add_request
                processed_inputs = {"prompt":xxx, "prompt_token_ids": [2, 31414, 略], "type":"token"}
                self._add_processed_request(request_id, processed_inputs, params=取样参数)
                    创建 seq: Sequence
                    创建 seq_group: SequenceGroup
                    seq_group 添加到 scheduler(最短未finish的)中的waiting中
                    返回 seq_group


    # 第二步, step, 单步推理 one decoding
    LLM._run_engine()
        while self.llm_engine.has_unfinished_requests():
            step_output = self.llm_engine.step()
                LLMEngine.step()
                    step1: 找到要处理的 sequences. 调用 scheduler.schedule()
                    step2: 使用 executor, executor the model
                    step3: 处理 model 输出
                    step4: 返回 最新生成的 结果

=============
Scheduler.schedule() 细节
    Scheduler._schedule()
        Scheduler._schedule_default() # 另外一条路是 chunked prefill
            _schedule_default 优先 prefill, 然后 decodes
                Scheduler._schedule_prefills
                    1. _schedule_会稍微delay, 等待 queue fill up
                    2. 检测是否还有足够的显存
                        需要的block = num_lookahead_slots + len(seq_token_ids) // block_size
                        free_block = 记录的free block
                    Scheduler.block_manager.can_allocate(seq_group, num_lookahead_slots)
                    Scheduler.waiting.popleft()
                    Scheduler._allocate_and_set_running(seq_group)
                        1. 给 seq_group分配 block
                        Scheduler.block_manager.allocate(seq_group)
                            BlockManager.allocate(seq_group)
                                waiting_seq:[] = seq_group中的waiting状态的seq
                                seq0= waiting_seq[0]
                                block_table: BlockTable = self._allocate_sequence(seq0) # 得到一个BlockTable, 还未分配
                                    block_table.allocate(token_ids=seq.token_ids, ...)
                                        BlockTable.allocate(token_ids, device=GPU, )
                                            BlockTable._allocate_blocks_for_token_ids(token_ids, device, )
                                                1. 对token_ids进行切块, 块大小=block_size, (最后一个块可能不足 block_size)
                                                   [ [第一块 token_ids,满], ... [最后一块, 不满块] ]
                                                2. 切块的token_ids, 按照顺序处理 => 得到 NaiveBlock
                                                    1.满块:
                                                        直接分配一个 immutable_block, 把 N个token_ids放进去
                                                    2. 非满的
                                                        TableBlock._allocator.allocate_mutable_block(...)
                                                        -> CpuGpuBlockAllocator._allocators[Device.GPU].allocate_mutable_block
                                                        -> NativeBlockAllocator.allocate_mutable_block
                                                            physical_block_id = [free队列_idx, 如 0,1,2,3, ...] pop left
                                                            NativeBlockAllocator._block_pool.init_block(prev_block, physical_block_id,)
                                                                pool_id = _free_ids.popleft() # pool_id 可以看出 block在pool中的序号
                                                                block = _block[pool_id] # 可以看到这个block在 _pool中
                                                        得到 block后, 把 N个token_ids放到 block中。NaiveBlock._token_ids:[]
                                                3. BlockTable.update(blocks=上面生成所有block)
                        2. 所有分配过 block的seq状态改为 running
                        [seq.status = Running for seq in seq_group]
