# 大流程

LLMEngine.step()
    output1 = engine_core_client.get_output()
        EngineCore.step()
            schedule_output = scheduler.schedule()
            model_output = execute_model(schedule_output)
    output2 = process_output(output1)
    return output2


