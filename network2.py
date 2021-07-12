"""

Network

Changed:
    - ResNet to be fully pre-activated
    - Change random seed TODO FUTURE WORK
    - Added additional convolutional layers to Policy and

"""

import os
import json
import numpy as np

import multiprocessing as mp


from api import API
from config import Config

from tensorflow import get_default_graph
from keras.regularizers import l2
from keras.layers.merge import Add
from keras.engine.training import Model
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.normalization import BatchNormalization


class Network:
    def __init__(self, config: Config):
        self.config = config
        self.pipes = []
        self.model = None
        self.graph = None
        self.api = None

    def create_pipes(self):
        if self.api is None:
            self.api = API(self)
            self.api.start_inference_worker()
        return [self.api.get_pipe() for _ in range(self.config.num_threads)]

    def build(self):
        """
        The neural network consist of a "body" followed by both policy and value "heads".
        The body consists of a rectified batch_normalized convolutional layer followed by 19 residual blocks.
        Each such block consists of two rectified batch-normalized convolutional layers with a skip connection.
        Each convolution applies 256 filters of kernel size 3x3 with stride 1.
        The policy head applies and additional rectified, batch-normalized convolutional layer,
        followed by a final convolution of 73 filters, representing the logits of the respective policies.
        The value head applies an additional rectified, batch-normalized convolution of 1 filter of kernel size 1x1
        with stride 1, followed by a rectified linear layer of size 256 and a tanh-linear layer of size 1.
        """
        filter_num = self.config.cnn_filter_num  # 256
        filter_size = self.config.cnn_filter_size  # 3
        residual_num = self.config.res_layer_num  # 19
        l2_reg = self.config.l2_reg  # 1e-4
        value_head_size = self.config.value_fc_size  # 256
        depth = self.config.t_history  # 8
        input_stack_height = 7 + depth * 14  # 119
        pieces = 12
        unique_pieces = 6

        in_x = x = Input((input_stack_height, 8, 8))
        x = Conv2D(filters=filter_num, kernel_size=filter_size, padding="same", data_format="channels_first",
                   use_bias=False, kernel_initializer="glorot_normal", bias_initializer="zeros",
                   kernel_regularizer=l2(l2_reg), input_shape=(input_stack_height, 8, 8))(x)

        for _ in range(residual_num):
            x = self._build_residual_block(x)

        # Policy Output
        res_out = x
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=256, kernel_size=3, data_format="channels_first", use_bias=False,
                   kernel_initializer="glorot_normal", bias_initializer="zeros", kernel_regularizer=l2(l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters=73, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_initializer="glorot_normal", bias_initializer="zeros", kernel_regularizer=l2(l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        x = Flatten()(x)
        policy_out = Dense(self.config.n_labels, kernel_initializer="glorot_normal", bias_initializer="zeros",
                           kernel_regularizer=l2(l2_reg), activation="softmax", name="policy_out")(x)

        # Value Output
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=1, kernel_size=1, data_format="channels_first", use_bias=False,
                   kernel_initializer="glorot_normal", bias_initializer="zeros", kernel_regularizer=l2(l2_reg))(res_out)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(value_head_size, kernel_initializer="glorot_normal", bias_initializer="zeros",
                  kernel_regularizer=l2(l2_reg), activation="relu")(x)
        value_out = Dense(1, kernel_initializer="glorot_normal", bias_initializer="zeros",
                          kernel_regularizer=l2(l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="alphaZeroClone")

    def _build_residual_block(self, x):
        filter_num = self.config.cnn_filter_num  # 256
        filter_size = self.config.cnn_filter_size  # 3
        l2_reg = self.config.l2_reg  # 1e-4

        in_x = x
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filter_num, kernel_size=filter_size, padding="same", data_format="channels_first",
                   use_bias=False, kernel_initializer="glorot_normal", bias_initializer="zeros",
                   kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)  # no issue since it is not the final activation
        x = Conv2D(filters=filter_num, kernel_size=filter_size, padding="same", data_format="channels_first",
                   use_bias=False, kernel_initializer="glorot_normal", bias_initializer="zeros",
                   kernel_regularizer=l2(l2_reg))(x)
        # add last according to https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
        x = Add()([in_x, x])
        return x

    def inference(self, image):
        # Get policy and value from network
        data = np.asarray(image, dtype=np.float32)
        policy, value = self.model.predict_on_batch(data)
        return policy, value

    def load(self, path_config, path_weight):
        if not os.path.exists(path_config):
            return False
        if not os.path.exists(path_weight):
            return False

        with open(path_config, "r") as f:
            self.model = Model.from_config(json.load(f))
        self.model.load_weights(path_weight)
        self.graph = get_default_graph()
        return True

    def save(self, path_config, path_weight):
        with open(path_config, "w") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(path_weight)
