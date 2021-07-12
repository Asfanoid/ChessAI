import numpy
import multiprocessing as mp

from multiprocessing import connection
from threading import Thread

from config import Config
from env import Result


class ReplayBuffer(object):

    def __init__(self, config: Config):
        self.config = config
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.pipes = []

    def create_pipes(self):
        return [self.get_pipe() for _ in range(self.config.num_actors + 1)]  # +1 for training process

    def get_pipe(self):
        a, b = mp.Pipe()
        self.pipes.append(a)
        return b

    def start(self):
        worker = Thread(target=self.replay_worker, name="replay_worker")
        worker.daemon = True
        worker.start()

    def replay_worker(self):
        while True:
            ready = mp.connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            print("HEI")
            data = []
            for pipe in ready:
                while pipe.poll():
                    data.append(pipe.recv())
            if not data:
                continue

            self.save_game(data)

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, buffer):
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in buffer))
        games = numpy.random.choice(
            buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in buffer])
        game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
        # print("sampling batch. move_sum:", move_sum)
        return [(g.make_target_image(i), g.make_target(i)) for (g, i) in game_pos]


class GameBuffer:
    def __init__(self, history, child_visits, images, result):
        self.history = history
        self.child_visits = child_visits
        self.images = images
        self.result = result

    def make_target_image(self, state_index: int):
        return self.images[state_index]

    def make_target(self, state_index: int):
        return (self.child_visits[state_index],
                self.terminal_value(state_index % 2))

    def terminal_value(self, to_play):
        modulus = to_play % 2
        if self.result == Result.WHITE:
            if modulus == 0:
                return -1
            else:
                return 1
        if self.result == Result.BLACK:
            if modulus == 0:
                return 1
            else:
                return -1
        if self.result == Result.DRAW:
            return 0
