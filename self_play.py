"""

SELF_PLAY


"""

import os
import time
import json
import numpy
import chess
import chess.pgn
import tensorflow as tf
import keras.backend as K

from trainer import TrainingModel
from network import Network
from logging import getLogger
from datetime import datetime
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
from helpers import load_network_after_training, save_network, make_network

from env import ChessEnv, MyBoard
from config import Config
from player import ChessPlayer

log = getLogger(__name__)


def start(config: Config):
    tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config.vram_frac,
                                                         allow_growth=False))

    sess = tf.Session(config=tf_config)
    K.set_session(sess)
    # from tensorflow.python.client import device_lib
    # print("my devices:", [x.name for x in device_lib.list_local_devices()])
    return SelfPlay(config).start()


class ReplayBuffer:
    def __init__(self, config: Config):
        self.config = config
        self.window_size = self.config.window_size
        self.batch_size = self.config.batch_size
        self.buffer = []
        self.dataset = []

    def save_to_file(self, white, black):
        data = []
        for moves in zip(white, black):
            for move in moves:
                data.append(move)
        if len(white) > len(black):
            data.append(white[-1])
        data_dir = self.config.resource.play_data_dir
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        save_path = os.path.join(data_dir, "play_data_" + date + ".json")

        # Might want to delete older games eventually
        with open(save_path, "w") as file:
            json.dump(data, file)

        log.info(f"Saved games to {save_path}")

    def save_to_buffer(self, white, black):
        data = []
        for moves in zip(white, black):
            for move in moves:
                data.append(move)
        if len(white) > len(black):
            data.append(white[-1])

        if len(self.buffer) > self.window_size:
            print("POP BUFFER", len(self.buffer))
            self.buffer.pop(0)

        # print("buffer length before:", len(self.buffer))
        self.buffer.append(data)
        # print("buffer length after:", len(self.buffer))

    def sample_batch(self):
        """ Is there an advantage to random sampling?

        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = numpy.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer]
        )
        game_pos = [(g, numpy.random.randint(len(g.moves))) for g in games]
        return [(g.make_image(i), g_make_target(i)) for (g, i) in game_pos)]

        """

        image_list = []
        policy_list = []
        value_list = []
        # print("buffer:", self.buffer)
        for games in self.buffer:
            # print("games:", games)
            board = MyBoard()
            for move, policy, value in games:
                # print("move", move)
                # print("value", value)
                image = board.make_image(self.config.t_history)
                if board.turn == chess.BLACK:
                    policy = self.config.change_pov(policy)

                image_list.append(image)
                policy_list.append(policy)
                value_list.append(value)

        return image_list, policy_list, value_list

    def collect_samples(self):
        if not self.buffer:
            return
        image_ary_list, policy_ary_list, value_ary_list = self.sample_batch()

        state_ary = numpy.stack(image_ary_list)
        policy_ary = numpy.stack(policy_ary_list)
        value_ary = numpy.expand_dims(numpy.stack(value_ary_list), axis=1)
        self.buffer = []
        return state_ary, policy_ary, value_ary


def update_weights(optimizer: tf.train.Optimizer, network: Network, image_batch, policy_batch, value_batch, weight_decay: float, pipes):
    loss = 0

    for image, target_policy, target_value in zip(image_batch, policy_batch, value_batch):
        policy_logits, value = network.inference(image)
        loss += (tf.losses.mean_squared_error(value, target_value) + tf.nn.softmax_cross_entropy_with_logits(
                    logits=policy_logits, labels=target_policy))

    for weights in network.model.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)


def train_network(config: Config, network: Network, replay_buffer: ReplayBuffer, pipes):
    # network = Network()
    # load_network(config, network)
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule, config.momentum)

    image, policy, value = replay_buffer.sample_batch()
    update_weights(optimizer, network, image, policy, value, config.weight_decay, pipes)
    save_network(config, network)


class SelfPlay:
    def __init__(self, config: Config):
        self.config = config
        self.replay_buffer = ReplayBuffer(config)
        self.network = make_network(config, name="model_")
        self.manager = Manager()
        self.pipes = self.manager.list([self.network.create_pipes() for _ in range(self.config.num_actors)])
        self.training_model = TrainingModel(self.config)

    def start(self):
        # start_training_worker()
        while True:
            """ Use this instead if you want to debug or watch games in terminal. Also add render."""
            # play_thread(self.config, self.replay_buffer, self.pipes)
            # time.sleep(2)  # Get chance to see score
            start_time = time.time()
            with ProcessPoolExecutor(max_workers=self.config.num_actors) as executor:
                for game in as_completed([executor.submit(play_thread, self.config, self.pipes)
                                          for _ in range(self.config.max_games_in_buffer)]):
                    white, black, result = game.result()
                    self.replay_buffer.save_to_buffer(white, black)
                    total_time = time.time() - start_time
                    # replay_buffer.save_to_file(white.data, black.data)
                    log.info(f"Time: {total_time:.2f}, {result}.")

            dataset = self.replay_buffer.collect_samples()
            self.training_model.start(dataset)
            # train_network(self.config, self.network, self.replay_buffer, self.pipes)
            load_network_after_training(self.config, self.network)


def play_thread(config: Config, pipes):
    start_time = time.time()
    pipe = pipes.pop()

    env = ChessEnv(config).reset()
    white = ChessPlayer(config,  pipe)
    black = ChessPlayer(config,  pipe)

    while not env.done:
        if env.board.turn == chess.WHITE:
            move = white.action(env, random_move=False)
        else:
            move = black.action(env, random_move=False)
        env.step(move)
        env.render()

    winner = env.get_result()
    white.add_result(winner)
    black.add_result(winner * -1)

    # START DEBUG'
    total_time = time.time() - start_time
    # print("%.4f" % total_time, env.result)
    #time.sleep(5)
    #env.board.make_image(8)
    #print("pgn start")
    pgn = chess.pgn.Game.from_board(env.board)
    print(pgn)
    #print("pgn done")
    #if len(env.board.move_stack) < 20:
        #print(pgn)  # The short games are usually the most interesting
    # END DEBUG

    pipes.append(pipe)
    return white.data, black.data, env.result  #, total_time

