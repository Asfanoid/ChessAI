"""

PSEUDO


"""
import os
import time
import json
import numpy as np
import chess
import chess.pgn
import tensorflow as tf
import keras.backend as K

from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from replay import ReplayBuffer, GameBuffer
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import time, sleep
from trainer import TrainingModel
from network import Network
from logging import getLogger
from datetime import datetime
from multiprocessing import Manager, Process
from concurrent.futures import ProcessPoolExecutor, as_completed
from helpers import load_network, alpha_load_network, save_network, make_network

from env import ChessEnv, MyBoard
from config import Config
from player import ChessPlayer


import chess
import numpy

from board import MyBoard
from env import Result
from alphazero_config import AlphaZeroConfig


def start(config: Config):
    """
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config.vram_frac,
                                                         allow_growth=False,))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)
    """
    # from tensorflow.python.client import device_lib
    # print("my devices:", [x.name for x in device_lib.list_local_devices()])
    # tf.enable_eager_execution()
    return AlphaZeroClone(config).start()


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Game(object):
    def __init__(self, config: Config, history=None):
        self.config = config
        self.history = history or []
        self.child_visits = []
        self.num_actions = 4672
        self.labels = config.labels

        self.board = MyBoard()
        self.result = None
        self.resigned = False
        self.images = []

    def get_result(self):
        return self.result

    def terminal(self):
        return self.result is not None

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

    def legal_actions(self):
        return list(self.board.generate_legal_moves())

    def clone(self):
        return Game(self.config, list(self.history))

    def apply(self, action):
        if action == chess.Move.null() or action is None:
            self.resign()
            return

        self.board.push(action)
        self.history.append(self.board.fen())

        if self.board.is_game_over() or \
                self.board.can_claim_draw() or \
                self.board.fullmove_number >= self.config.max_moves:
            result = self.board.result()
            if result == "1/2-1/2" or \
                    self.board.can_claim_draw() or \
                    self.board.fullmove_number >= self.config.max_moves:
                self.result = Result.DRAW
            else:
                if result == "1-0":
                    self.result = Result.WHITE
                if result == "0-1":
                    self.result = Result.BLACK

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        # self.child_visits.append([
            # root.children[a].visit_count / sum_visits if a in root.children else 0
            # for a in self.labels])
        policy = np.zeros(self.num_actions)
        policy[[self.labels[a] for a in root.children.keys()]] = \
            [a.visit_count / sum_visits for a in root.children.values()]
        self.child_visits.append(policy)

    def make_image(self):
        image = self.board.gather_features(self.config.t_history)
        self.images.append(image)
        return image

    def make_target_image(self, state_index: int):
        return self.images[state_index]

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2

    def resign(self):
        self.resigned = True
        print("RESIGNING")
        if self.board.turn == chess.WHITE:
            self.result = Result.BLACK
        else:
            self.result = Result.WHITE


def train_network_using_keras(config: Config, buffer: list):
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config.vram_frac,
                                                         allow_growth=False, ))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)
    sleep(15)

    network = make_network(config, name="alpha_")
    replay_buffer = ReplayBuffer(config)

    optimizer = SGD(config.learning_rate, config.momentum)
    losses = ['categorical_crossentropy', 'mean_squared_error']
    network.model.compile(optimizer=optimizer, loss=losses)
    # tb = TensorBoard(log_dir=config.resource.log_dir, histogram_freq=1)

    wait_for_buffer(config.batch_size, buffer)
    for i in range(config.training_steps):
        print("Training step:", i)
        if i % config.checkpoint_interval == 0:
            save_network(config, network, name="alpha_")

        batch = replay_buffer.sample_batch(get_games(buffer))
        image, policy, value = load_data(batch)
        network.model.fit(image, [policy, value], batch_size=config.batch_size, validation_split=0.05)

    save_network(config, network, name="alpha_")


def wait_for_buffer(batch_size, buffer):
    sleep(10)
    buffer_length = 0
    while buffer_length < batch_size:  # 4096
        print("Buffer is not ready.", buffer_length, "of", batch_size)
        sleep(60)
        move_sum = 0
        for game in buffer:
            move_sum += len(game.history)
        buffer_length = move_sum
    print("Buffer ready")


def load_data(batch):
    images, policys, values = [], [], []
    for image, (policy, value) in batch:
        images.append(image)
        policys.append(policy)
        values.append(value)
    return numpy.asarray(images), numpy.asarray(policys), numpy.asarray(values)


def train_network(config: Config, buffer):
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config.vram_frac,
                                                         allow_growth=False, ))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    network = make_network(config, name="alpha_")
    replay_buffer = ReplayBuffer(config)
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule, config.momentum)
    # Make tf variables before sess.run()     (loss, optimizer, and train)

    sleep(40)
    for i in range(config.training_steps):
        print("Training step:", i)
        if i % config.checkpoint_interval == 0:
            save_network(config, network, name="alpha_")
        batch = replay_buffer.sample_batch(get_games(buffer))
        update_weights(optimizer, network, batch, config.weight_decay)
    save_network(config, network)


def get_games(buffer):
    return [g for g in buffer]


def loss(value, target_value, policy, target_policy):
    return (tf.losses.mean_squared_error(value, target_value) + tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=policy, labels=target_policy))


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch, weight_decay: float):
    for image, (target_value, target_policy) in batch:
        with tf.GradientTape as tape:
            policy, value = network.inference(image)
            target_policy = numpy.asarray(target_policy, dtype=numpy.float32)
            value = float(value[0])
            target_value = float(target_value)

            loss = loss(value, target_value, policy, target_policy)

        grads = tape.gradient(loss, network.model.variables)
        optimizer.apply_gradients(zip(grads, network.model.variables))

    for weights in network.model.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    vars = tf.trainable_variables()  # debug
    grads = optimizer.compute_gradients(loss)
    train = optimizer.minimize(loss)
    train.run()


def predict(image, pipes):
    pipe = pipes.pop()
    pipe.send(image)
    ret = pipe.recv()
    pipes.append(pipe)
    return ret


class AlphaZeroClone:
    def __init__(self, config: Config):
        self.config = config
        self.replay_buffer = ReplayBuffer(config)
        self.manager = Manager()
        # self.pipes = self.manager.list(self.replay_buffer.create_pipes())
        self.buffer = self.manager.list()
        # self.network = make_network(config, name="model_")
        # self.training_model = make_network(config, name="training_")

    def start(self):
        for i in range(self.config.num_actors):
            self.launch_job(self.config, self.buffer)

        # train_network(self.config, self.buffer)
        train_network_using_keras(self.config, self.buffer)

        """g
        for i in range(100):
            sleep(20)
            game = self.buffer.pop()
            print(game.history[5])
        print("All done.")

        
        with ProcessPoolExecutor(max_workers=self.config.num_actors) as executor:
            for game in as_completed([executor.submit(run_selfplay, self.config, self.pipes)
                                      for _ in range(self.config.num_actors)]):
                print("Done.")
        """

    def launch_job(self, config: Config, buffer):
        process = Process(target=run_selfplay, args=(config, buffer))
        process.daemon = True
        process.start()


def run_selfplay(config: Config, buffer):
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=config.vram_frac,
                                                         allow_growth=True, ))
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    network = make_network(config, name="alpha_")
    # pipe = pipes.pop()
    while True:
        alpha_load_network(config, network, name="alpha_")
        game = play_game(config, network)
        pgn = chess.pgn.Game.from_board(game.board)
        print(pgn)
        add_to_buffer(game, config.window_size, buffer)


def add_to_buffer(game, window_size, buffer):
    if len(buffer) > window_size:
        buffer.pop(0)
    buffer.append(GameBuffer(game.history, game.child_visits, game.images, game.result))


def play_game(config: Config, network: Network):
    game = Game(config)
    start_time = time()
    while not game.terminal():
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
    total_time = time() - start_time
    print("%.3f" % total_time, game.result)
    return game


def run_mcts(config: Config, game: Game, network: Network):
    root = Node(0)
    evaluate(root, game, network, config)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        value = evaluate(node, scratch_game, network, config)
        backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root


def select_action(config: Config, game: Game, root: Node):
    visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max_sample(visit_counts)
    return action


def softmax_sample(visit_counts):
    policy = numpy.zeros(len(visit_counts))
    total_visits = 0
    for i, visits in enumerate(visit_counts):
        # print("i", i, "visit:", visits[0], "move", str(visits[1]))
        policy[i] = visits[0]
        total_visits += visits[0]
    policy = policy / total_visits
    index = int(numpy.random.choice(range(len(visit_counts)), p=policy))
    # print("soft:", visit_counts[index])
    return visit_counts[index]


def max_sample(visit_counts):
    policy = numpy.zeros(len(visit_counts))
    for i, visits in enumerate(visit_counts):
        policy[i] = visits[0]
    best_move = numpy.argmax(policy)
    return visit_counts[best_move]


def select_child(config: Config, node: Node):
    nodes = defaultdict(str)
    for action, child in node.children.items():
        nodes[action] = ucb_score(config, node, child)

    # _, action, child = max([ucb_score(config, node, child), action, child] for action, child in node.children.items())
    action = max(nodes, key=nodes.get)
    for move, child in node.children.items():
        if move == action:
            return move, child


def ucb_score(config: Config, parent: Node, child: Node):
    pb_c = numpy.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
    pb_c *= numpy.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


def evaluate(node: Node, game: Game, network, config):
    image = game.make_image()
    policy_logits, value = network.inference(image)

    # Expand the node.
    node.to_play = game.to_play()
    policy = {a: numpy.exp(policy_logits[config.labels[a]]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value


def backpropagate(search_path: [Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1


def add_exploration_noise(config: Config, node: Node):
    actions = node.children.keys()
    noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
