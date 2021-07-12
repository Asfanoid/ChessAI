import numpy as np
import multiprocessing as mp

from multiprocessing import connection
from threading import Thread
from time import time


class API:
    def __init__(self, model):
        self.model = model
        self.pipes = []

    def start_inference_worker(self):
        worker = Thread(target=self.inference_worker, name="inference_batch_worker")
        worker.daemon = True
        worker.start()

    def get_pipe(self):
        a, b = mp.Pipe()
        self.pipes.append(a)
        return b

    def inference_worker(self):
        while True:
            ready = mp.connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
                    result_pipes.append(pipe)
            if not data:
                continue
            # time_numpy = time()
            data = np.asarray(data, dtype=np.float32)
            # print("numpy data array:", time() - time_numpy)
            # time_predict = time()
            with self.model.graph.as_default():
                policy, value = self.model.model.predict_on_batch(data)
            # print("predict:", time() - time_predict)
            # time_send = time()
            for pipe, p, v in zip(result_pipes, policy, value):
                pipe.send((p, float(v)))
            # print("send:", time() - time_send)
