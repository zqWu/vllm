def get_single_gpu():
    """获取最大可用显存的gpu"""
    import GPUtil
    gpus = GPUtil.getGPUs()

    gpu_id = None
    max_free_mem = 0
    total_mem = 0
    for gpu in gpus:
        avail_mem = gpu.memoryFree
        if avail_mem > max_free_mem:
            gpu_id = gpu.id
            total_mem = gpu.memoryTotal
            max_free_mem = avail_mem

    free_mem_ratio = max_free_mem / total_mem
    print(f"使用 gpu:{gpu_id}, 剩余显存 {max_free_mem / 1024:.2f}G, 占 {free_mem_ratio * 100.0:.3f}%")
    return gpu_id, max_free_mem, free_mem_ratio


def print_pid_tid():
    import os, threading
    print(f"PID: {os.getpid()}, tid={threading.get_ident()}")
