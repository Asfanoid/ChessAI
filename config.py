"""

Configs for everyone!
Mostly taken from Benediamond's git repos

https://github.com/benediamond/chess-alpha-zero

"""


import os
import math
import chess
import numpy as np


def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_data_dir():
    return os.path.join(get_project_dir(), "data")


class Config:
    def __init__(self, config_type="normal"):
        # self.human = PlayWithHuman()
        self.resource = ResourceConfig()
        self.evaluate = EvaluateConfig()
        self.trainer = TrainerConfig()
        self.model = ModelConfig()
        self.labels = create_uci_labels()

        ############################
        # Alpha Zero Configuration #
        ############################

        # Self play
        self.num_actors = 12  # was 5000, but i guess this is number of processes, so we must change this
        self.num_sampling_moves = 30  # Number of moves with exploration noise
        self.max_moves = 512
        self.num_simulations = 800

        # Noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training
        self.training_steps = 700000
        self.checkpoint_interval = 1000
        self.window_size = 100  # Max number of games in buffer (Should not exceed RAM!)
        self.batch_size = 4096

        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.learning_rate = 0.02
        self.learning_rate_schedule = {
            0: 0.2,
            100000: 0.002,
            300000: 0.0002,
            500000: 0.00002
        }

        #####################
        # Own Configuration #
        #####################
        self.max_games_in_buffer = 100  # How much RAM do we have to store self played games?
        self.max_file_num = 400

        self.num_threads = 16
        self.vram_frac = 1.0

        self.n_labels = 4672  # 64*73
        self.c_puct = 3  # Are we using this?

        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 19
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.t_history = 8
        self.input_stack_height = 7 + self.t_history * 14

        if config_type == "delivery":
            self.num_simulations = 800
            self.num_actors = 1
            self.num_threads = 16
            self.max_moves = 3
            self.vram_frac_player = 0.4
            self.vram_frac_trainer = 0.25

            self.cnn_filter_num = 16
            self.res_layer_num = 1
            self.value_fc_size = 16
            self.t_history = 1
            self.input_stack_height = 7 + self.t_history * 14

            self.window_size = 40  # Max number of games in buffer (Should not exceed RAM!)
            self.batch_size = 99999

            self.num_sampling_moves = 0
            self.root_dirichlet_alpha = 0
            self.root_exploration_fraction = 0

        if config_type == "home":
            self.num_simulations = 200
            self.num_actors = 10
            self.num_threads = 16
            self.max_moves = 128
            self.vram_frac_player = 0.07
            self.vram_frac_trainer = 0.25

            self.cnn_filter_num = 16
            self.res_layer_num = 1
            self.value_fc_size = 16
            self.t_history = 1
            self.input_stack_height = 7 + self.t_history * 14

            self.window_size = 40  # Max number of games in buffer (Should not exceed RAM!)
            self.batch_size = 4096

        if config_type == "school":
            self.num_simulations = 128
            self.num_actors = 1
            self.num_threads = 16
            self.max_moves = 128
            self.vram_frac = 0.07

            self.cnn_filter_num = 256
            self.res_layer_num = 1
            self.value_fc_size = 256
            self.t_history = 1
            self.input_stack_height = 7 + self.t_history * 14

            self.window_size = 40  # Max number of games in buffer (Should not exceed RAM!)
            self.batch_size = 4096

        if config_type == "test":
            self.num_simulations = 40
            self.num_actors = 1
            self.num_threads = 16
            self.max_moves = 32
            self.max_games_in_buffer = 10
            self.vram_frac = 0.1

            self.cnn_filter_num = 256
            self.res_layer_num = 1
            self.value_fc_size = 256
            self.t_history = 1
            self.input_stack_height = 7 + self.t_history * 14

            self.window_size = 40  # Max number of games in buffer (Should not exceed RAM!)
            self.batch_size = 256  # 4096

    # Taken from Benediamond's Repository
    def change_pov(self, leaf):
        new = np.zeros(self.n_labels)
        for f in range(8):
            for r in range(8):
                for v in range(73):
                    if v in range(56):
                        block = v // 14
                        position = v % 14
                        if position >= 7:
                            new_position = position - 7
                        else:
                            new_position = position + 7
                        new_v = block * 14 + new_position
                    elif v in range(56, 64):
                        if v % 2 == 0:
                            new_v = v + 1
                        else:
                            new_v = v - 1
                    else:
                        new_v = v
                    new[v * 64 + r * 8 + f] = leaf[new_v * 64 + (7 - r) * 8 + (7 - f)]
        return list(new)


class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", get_project_dir())
        self.data_dir = os.environ.get("DATA_DIR", get_data_dir())
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.old_model_dir = os.path.join(self.model_dir, "old_models")
        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

        self.model_dirname_tmpl = "model_%s"
        self.model_config_filename = "model_config.json"
        self.model_weight_filename = "model_weight.h5"
        self.play_data_filename_tmpl = "play_%s.json"

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir, self.old_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class EvaluateConfig:
    def __init__(self):
        self.Replace_rate = 0.55
        self.game_num = 100
        self.nb_game_in_file = 100
        self.max_file_num = 400
        self.simulations_per_move = 200
        self.noise_eps = 0


class TrainerConfig:
    def __init__(self):
        self.batch_size = 32
        self.cleaning_processes = 8
        self.vram_frac = 1.0
        self.epoch_to_checkpoint = 1
        self.start_total_steps = 0
        self.save_model_steps = 10000
        self.load_data_steps = 1000
        self.min_data_size_to_learn = 10000
        self.max_num_files_in_memory = 20


class ModelConfig:
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 19
        self.l2_reg = 1e-4
        self.value_fc_size = 256
        self.t_history = 8
        self.input_stack_height = 7 + self.t_history*14


def create_uci_labels():
    labels = {}
    for f in range(8):
        for r in range(8):
            for v in range(0, 7):
                f_new = f + (v + 1)
                _add_move(labels, v, f, r, f_new, r)
            for v in range(7, 14):
                f_new = f - (v - 6)
                _add_move(labels, v, f, r, f_new, r)
            for v in range(14, 21):
                r_new = r + (v - 13)
                _add_move(labels, v, f, r, f, r_new)
            for v in range(21, 28):
                r_new = r - (v - 20)
                _add_move(labels, v, f, r, f, r_new)
            for v in range(28, 35):
                f_new = f + (v - 27)
                r_new = r + (v - 27)
                _add_move(labels, v, f, r, f_new, r_new)
            for v in range(35, 42):
                f_new = f - (v - 34)
                r_new = r - (v - 34)
                _add_move(labels, v, f, r, f_new, r_new)
            for v in range(42, 49):
                f_new = f + (v - 41)
                r_new = r - (v - 41)
                _add_move(labels, v, f, r, f_new, r_new)
            for v in range(49, 56):
                f_new = f - (v - 48)
                r_new = r + (v - 48)
                _add_move(labels, v, f, r, f_new, r_new)
            _add_move(labels, 56, f, r, f + 2, r + 1)
            _add_move(labels, 57, f, r, f - 2, r - 1)
            _add_move(labels, 58, f, r, f + 1, r + 2)
            _add_move(labels, 59, f, r, f - 1, r - 2)
            _add_move(labels, 60, f, r, f + 2, r - 1)
            _add_move(labels, 61, f, r, f - 2, r + 1)
            _add_move(labels, 62, f, r, f + 1, r - 2)
            _add_move(labels, 63, f, r, f - 1, r + 2)
            if r == 6:
                _add_move(labels, 64, f, r, f, r + 1, 4)
                _add_move(labels, 65, f, r, f, r + 1, 3)
                _add_move(labels, 66, f, r, f, r + 1, 2)
                _add_move(labels, 67, f, r, f + 1, r + 1, 4)
                _add_move(labels, 68, f, r, f + 1, r + 1, 3)
                _add_move(labels, 69, f, r, f + 1, r + 1, 2)
                _add_move(labels, 70, f, r, f - 1, r + 1, 4)
                _add_move(labels, 71, f, r, f - 1, r + 1, 3)
                _add_move(labels, 72, f, r, f - 1, r + 1, 2)
            elif r == 1:
                _add_move(labels, 64, f, r, f, r - 1, 4)
                _add_move(labels, 65, f, r, f, r - 1, 3)
                _add_move(labels, 66, f, r, f, r - 1, 2)
                _add_move(labels, 67, f, r, f - 1, r - 1, 4)
                _add_move(labels, 68, f, r, f - 1, r - 1, 3)
                _add_move(labels, 69, f, r, f - 1, r - 1, 2)
                _add_move(labels, 70, f, r, f + 1, r - 1, 4)
                _add_move(labels, 71, f, r, f + 1, r - 1, 3)
                _add_move(labels, 72, f, r, f + 1, r - 1, 2)
    return labels


def _add_move(labels, v, f, r, f_new, r_new, promotion=None):
    if f_new in range(0, 8) and r_new in range(0, 8):
        labels[chess.Move(r * 8 + f, r_new * 8 + f_new, promotion)] = v * 64 + r * 8 + f
        if promotion is None and (r == 6 and r_new == 7 and abs(f_new - f) <= 1 or r == 1
                                  and r_new == 0 and abs(f_new - f) <= 1):
            labels[chess.Move(r * 8 + f, r_new * 8 + f_new, 5)] = v * 64 + r * 8 + f  # add a default queen promotion.
