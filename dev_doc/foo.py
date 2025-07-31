import pickle
import random
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Pipe, Process

import zmq
from zmq import XPUB_VERBOSE

_ = """
# 写一个程序，实现以下功能:

1. 有一个主进程，2个子进程
2. 主进程启动时行为如下
    1. 启动一个 zmq, bind到:9999, 使用XPUB模式
    2. 启动子进程, 参数=[主进程zmq的socket地址, Pipe()创建的管道], 主进程持有Pipe的一端

3. 子进程启动后行为:
    1. 根据主进程的 socket, 连接上主进程 zmq-XPUB
    2. 自己启动一个 zmq/XPUB, bind到某个端口 [10001-10010]
    3. pipe一端发送 {"status": b"ready", "worker_zmq_socket": 自身zmq地址}
    4. pipe关闭, 不再使用

4. 主进程pipe收到某个子进程后
    1. 得到 worker_zmq_socket, 连接到子进程的 zmq/XPUB
    2. 关闭 pipe, 不再使用

[以上是系统初始化, 下面是系统协同工作]

5. 主进程在自己的 zmq/XPUB中广播消息(如 "sum_all, [1,2,3]"), 等待子进程消息(子进程的 zmq/XPUB)
6. 子进程收到广播消息, 进行处理
7. 子进程处理完毕后, 发送消息
8. 主进程等待,收到所有子进程消息, 处理完毕

# 设计
1. 使用 pickle.dumps/loads 进行 serde
2. 增加 collective_rpc

"""


def next_bind_port(start=10001, end=10010):
    """
    这段代码在并行执行时可能有问题
    A 进程 返回 10001, 但是没有迅速 listen
    导致 B 进程也返回 10001
    """
    import socket
    for port in range(start, end + 1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # ipv4
        time.sleep(random.randint(0, 5))  # 加入随机事件抖动, 降低端口冲突可能
        try:
            s.bind(("", port))
            print(f"Reserved free port {port}")
            return port
        except OSError:
            s.close()
            continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


@dataclass
class Handle:
    socket_addr: str
    reader_list: list[str]


class MessageQueue:
    def __init__(self, socket_addr: str, reader_list: list[str | int], self_id: str):
        self._is_writer = True
        self.socket_addr = socket_addr
        self.reader_list = reader_list
        # create zmq/XPUB
        context = zmq.Context()
        socket = context.socket(zmq.XPUB)
        socket.setsockopt(XPUB_VERBOSE, True)
        socket.bind(self.socket_addr)

        self.socket = socket
        self._id = self_id
        print(f"[MQ][{self._id}] create writer @ {self.socket_addr}")

    def send(self, obj):
        print(f"[MQ][{self._id}] send: {obj}")
        _bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.socket.send(_bytes)

    def recv(self):
        _bytes = self.socket.recv()
        obj = pickle.loads(_bytes)
        print(f"[MQ][{self._id}] recv: {obj}")
        return obj

    def handle(self):
        return Handle(
            socket_addr=self.socket_addr,
            reader_list=self.reader_list
        )

    @staticmethod
    def create_by_handle(handle: Handle, self_id):
        self = MessageQueue.__new__(MessageQueue)
        self.socket_addr = handle.socket_addr
        self.reader_list = handle.reader_list
        self._is_writer = False
        # connect to zmq/XPUB
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.socket_addr)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket = socket
        self._id = self_id
        #
        print(f"[MQ][{self._id}] create reader to {self.socket_addr}")
        return self

    def wait_ready(self):
        if self._is_writer:
            # 等待全部 reader连接上来, 然后发送 "ready"
            for i in range(len(self.reader_list)):
                # wait for subscription messages from all local readers
                self.socket.recv()  # 仅在这里使用 socket.recv(), 收 订阅信息 0b\x1
            self.send({"status": "READY"})
        else:
            # 连接到 writer, 然后等待接收,  "ready"
            recv = self.recv()
            assert recv["status"] == "READY"


class Executor:
    def __init__(self, n_worker: int = 2):
        self._n_worker = n_worker
        self._workers = {idx: {} for idx in range(n_worker)}
        self._id = "executor"
        self._mq_socket = 'tcp://127.0.0.1:5555'

        self.rpc_broadcast_mq = MessageQueue(
            socket_addr=self._mq_socket,
            reader_list=[_id for _id in self._workers.keys()],
            self_id=self._id,
        )
        self.handle = self.rpc_broadcast_mq.handle()
        self.job_list = []
        #
        self.init_worker()

        # check ready
        self.rpc_broadcast_mq.wait_ready()
        print(f"[{self._id}] done: rpc_broadcast_mq.wait_ready()")

        for worker_id, worker in self._workers.items():
            worker["worker_response_mq"].wait_ready()
            print(f"[{self._id}] done: worker_response_mq.wait_ready()")

        print(f"[{self._id}] ready")

    def init_worker(self):
        unready_worker = []
        for worker_id in self._workers.keys():
            reader, writer = Pipe(duplex=True)
            process = Process(target=Worker.spawn_worker, args=(self.handle, worker_id, writer))
            process.start()
            unready_worker.append({
                "worker_id": worker_id,
                "process": process,
                "reader": reader,
            })

        for ele in unready_worker:
            worker_id = ele["worker_id"]
            reader = ele["reader"]
            msg = reader.recv()
            assert msg["status"] == "READY"
            handle = msg["handle"]
            worker_response_mq = MessageQueue.create_by_handle(handle, self._id)
            self._workers[worker_id]["worker_response_mq"] = worker_response_mq
            reader.close()

    def collective_rpc(self, method_name, args, kwargs, response_worker_id=None):
        self.rpc_broadcast_mq.send((method_name, args, kwargs, response_worker_id))

        for _id, worker in self._workers.items():
            this_id_should_resp = response_worker_id is None or _id == response_worker_id
            if this_id_should_resp:
                obj = worker["worker_response_mq"].recv()
                print(f"[{self._id}] recv from [{_id}] [{obj}]")


class Worker:
    def __init__(self, handle: Handle, worker_id):
        self.handle = handle
        self._id = worker_id
        self.rpc_broadcast_mq = MessageQueue.create_by_handle(handle, worker_id)
        self.worker_response_mq = MessageQueue(
            socket_addr=f"tcp://localhost:{next_bind_port()}",
            reader_list=["1"],
            self_id=worker_id,
        )
        self.handle = self.worker_response_mq.handle()

    def wait_and_process(self):
        while True:
            method_name, args, kwargs, output_rank = self.rpc_broadcast_mq.recv()
            try:
                method = getattr(self, method_name)
                output = method(*args, **kwargs)
                if output_rank is None or self._id == output_rank:
                    self.worker_response_mq.send(output)
            except Exception:
                traceback.print_exc()
                self.worker_response_mq.send("TODO: 错误处理")

    def init_device(self, *args, **kwargs):
        time.sleep(random.randint(1, 3))
        return f"{self._id}: init_device done {args} {kwargs}"

    def say_hello(self, *args, **kwargs):
        time.sleep(random.randint(1, 3))
        return f"{self._id}: say_hello done {args} {kwargs}"

    @staticmethod
    def spawn_worker(handle: Handle, worker_id: str, writer) -> None:
        worker = Worker(handle, worker_id)
        # pipe 发送 ready
        writer.send({"status": "READY", "handle": worker.handle})
        writer.close()

        # wait mq ready
        worker.rpc_broadcast_mq.wait_ready()
        print(f"[{worker_id}] done: rpc_broadcast_mq.wait_ready()")

        worker.worker_response_mq.wait_ready()
        print(f"[{worker_id}] done: worker_response_mq.wait_ready()")
        #
        print(f"[{worker._id}] ready")
        worker.wait_and_process()


if __name__ == "__main__":
    executor = Executor(n_worker=2)
    time.sleep(2)

    print(f"=======collective_rpc:init_device==========")
    executor.collective_rpc("init_device", ['a', 1], {"foo": "bar"}, None)

    print(f"=======collective_rpc:say_hello==========")
    executor.collective_rpc("say_hello", [], {}, 1)
