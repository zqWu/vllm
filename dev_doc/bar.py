import threading
import uuid

import zmq


def make_zmq_socket(addr, bind: bool = False, socket_type=zmq.PUSH):
    context = zmq.Context()
    socket = context.socket(socket_type)
    if bind:
        print(f"bind {addr}")
        socket.bind(addr)
    else:
        print(f"connect {addr}")
        socket.connect(addr)
    return socket


class WorkerProc:
    def __init__(self, *args, **kwargs):
        self.worker_id = kwargs["worker_id"]
        self.input_addr = kwargs["input_addr"]
        self.input_socket = self.prepare_input_socket()

        threading.Thread(target=self.run_input_socket).start()

    def prepare_input_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        print(f"set identity to {self.worker_id}")
        socket.setsockopt(zmq.IDENTITY, str(self.worker_id).encode())  # 提前设置, 在 connect之前
        socket.connect(self.input_addr)
        return socket

    def run_input_socket(self):
        while True:
            _, msg = self.input_socket.recv_multipart()
            print(msg)
            self.input_socket.send_multipart([b'', msg])  # 收到 identity, b'', msg

    @staticmethod
    def run_in_process(worker_id, input_addr):
        kwargs = {
            "worker_id": worker_id,
            "input_addr": input_addr,
        }
        _ = WorkerProc(**kwargs)


class Client:
    def __init__(self, n_worker: int = 1):
        self.input_addr = f"ipc:///tmp/{uuid.uuid4()}"
        self.input_socket = make_zmq_socket(self.input_addr, bind=True, socket_type=zmq.ROUTER)

        self.workers = []
        for worker_idx in range(n_worker):
            self.make_worker(worker_idx)

    def process_request(self):
        print(f"===================")
        worker = self.workers[0]
        worker_id = str(worker["worker_id"]).encode()
        self.input_socket.send_multipart([worker_id, b'', b'hello'])  # 收到 b'', b'hello'

        # 等待响应
        worker_id = worker["worker_id"]
        msg = self.input_socket.recv_multipart()
        print(f"worker-[{worker_id}] msg={msg}")

    def make_worker(self, idx: int):
        worker_info = {
            "worker_id": idx,
            "input_addr": self.input_addr,
        }

        WorkerProc.run_in_process(idx, self.input_addr)
        self.workers.append(worker_info)


def main():
    client = Client()
    client.process_request()
    client.process_request()


if __name__ == "__main__":
    main()
    _ = """
DEALER - ROUTER

=== 结对1 ===
router: socket.send_multipart([identity, b'', msg]) # 发送时 有 identity
dealer: _, msg = socket.recv_multipart()            # 接收时 无 identity

=== 结对2 ===
dealer: socket.send_multipart([b'', msg])           # 发送时 无 identity
router: identity, _, msg = socket.recv_multipart()  # 收到时 有 identity

【助记】:指定id时要带, 自身id时zmq自动增/删

b'' 是zmq中的 delimiter frame. 省略会破坏兼容性
"""
