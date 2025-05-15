=============
vllm_config 包含了所有配置信息

engine:LLMEngine
    |__ model_executor:UniProcExecutor或其他
        |__ driver_worker:WorkerWrapperBase
            |__ worker:vllm.worker.worker.Worker
                |__ model_runner: GPUModelRunnerBase -> ModelRunner
                    |__ model: nnModule 具体的LLM类, 比如 OPTForCausalLM

=============

LLMEngine.from_engine_args
    找到 executor_class
        依据: engine_config.parallel_config
        默认是 uni, 也就是 单执行器
        有哪些Executor: 
            ExecutorWithExternalLauncher
            MultiprocExecutor
            RayDistributedExecutor
            UniProcExecutor
                DummyExecutor
            如果 tp=2 <======================== TODO
    初始化 self.tokenizer (最后调用 AutoTokenizer.from_pretrained(/path/to/llm/weight))
    初始化 self.detokenizer

    self.model_executor = executor_class(vllm_config)
        ExecutorBase.__init__ 仅记录一些config信息
            UniProcExecutor._init_executor
                初始化 self.driver_worker(vllmconfig, rpc_rank=0)
                    WorkerWrapperBase.__init__(...) 
                        记录 self.rpc_rank, vllmconfig

                self.collective_rpc("init_worker", args=([kwargs], ))
                    WorkerWrapperBase::init_worker
                        load_general_plugins() <======================== TODO
                        获取 worker_class = 'vllm.worker.worker.Worker'
                        self.worker = worker_class(...)
                            vllm.worker.worker.Worker.__init__
                                记录一些信息
                                is_driver_class=True <================== TODO
                                self.model_runner: GPUModelRunnerBase = ...
                                    GPUModelRunnerBase.__init__ <======= 最后干活的类
                                        性能有关: cuda_graph | kv_cache | profile | attn_backend
                                        load_model(...)
                                        profile_run(...)

                self.collective_rpc("init_device")
                    self.driver_worker.init_device = WorkerWrapperBase::init_device
                        WorkerWrapperBase.init_device()
                            vllm.worker.worker.Worker::init_device()
                                cuda时:
                                    torch.cuda.set_device(self.device)
                                    torch.cuda.empty_cache()
                                    ...
                                init_worker_distributed_environment 分布式环境初始化    

                self.collective_rpc("load_model")
                    driver_worker.load_model = WorkerWrapperBase::load_model
                        worker.load_model()
                            model_runner.load_model()
                                根据 load_config.load_format, 得到 loader. 如 DefaultModelLoader
                                loader.load_model(vllm_config), 这个整体上就是 pytorch load权重的方式
                                    类初始化 model = vllm.model_executor.models.OPTForCausalLM(...)
                                    model.load_weights(...)
                                    做一些 sanity check
                                    return model.eval()
                                model 加载完成, 分析model占用了内存 = (前值内存占用量 - 当前占用量)


对于 UniProcExecutor, Worker和 LLMEngine在同一个 process和 thread
