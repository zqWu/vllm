=============
vllm_config 包含了所有配置信息

engine:LLMEngine
    |__ model_executor:UniProcExecutor或其他
        |__ driver_worker:WorkerWrapperBase
            |__ worker:vllm.worker.worker.Worker
                |__ model_runner: GPUModelRunnerBase -> ModelRunner
                    |__ model: nnModule 具体的LLM类, 比如 OPTForCausalLM

=============

LLMEngine._initialize_kv_cache()
    model_executor.determine_num_available_blocks() <========= 检查 worker上能够分配多少个 kv_cache_block
        driver_worker.determine_num_available_blocks
            对每个 worker 都跑以下测试, 然后 取min操作
                worker.determine_num_available_blocks
                    # 准备工作
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.mem_get_info()
                    # 分析权重占用内存
                    memory_profiling(weights_memory=self.model_runner.model_memory_usage)
                    ModelRunner.profile_run()::GPUModelRunnerBase
                    ModelRunner._dummy_run(max_num_batched_tokens=2048, max_num_seqs=256)
                    ModelRunner.execute_model(model_input, kv_caches, intermediate_tensors) # 模拟一次推理
                        进行一次推理 OPTForCausalLM(mock_input, ...) => 单次 forward()
                        比对内存
                        可用的kv_cache内存 = gpu_mem * util% - non_cache_推理一次占用的内存
                        每个 kv_cache_block大小 = n_hidden_layer * block_size * (key大小 + valu大小) * 数据类型(如fp16=2bytes)
                        kv_cache block数量 = 可用的 kv_cahce内存 / (每个 kv cache block 大小)

    model_executor.initialize_cache()
        UniProcExecutor.initialize_cache()
            WorkerBase.initialize_cache()
                vllm.worker.worker.Worker.initialize_cache()
                    self._init_cache_engine()
                        self.cache_engine = [ CacheEngine() for _ in range(pp并行度)]
                    self._warm_up_model()