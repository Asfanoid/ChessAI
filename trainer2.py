import numpy as np

from multiprocessing import Process
from config import Config
from network import Network
from helpers import make_network, save_network

from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def start_training_worker():
    worker = Process(target=Training.start(), name="training_worker")


class TrainingModel:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
        self.model = make_network(self.config, name="training_")
        self.optimizer = None

    def start(self, dataset):
        # self.model = make_network(self.config)
        self.training(dataset)

    def training(self, dataset):
        self.compile_model()
        self.train_epoch(dataset)
        save_network(self.config, self.model, name="training_")

    def train_epoch(self, dataset):
        batch_size = 32  # 2048
        epochs = 1
        state_deque, policy_deque, value_deque = dataset
        state_ary, policy_ary, value_ary = np.asarray(state_deque), np.asarray(policy_deque), np.asarray(value_deque)
        tensorboard_cb = TensorBoard(log_dir=self.config.resource.log_dir, batch_size=batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary], batch_size=batch_size, epochs=epochs, shuffle=True,
                             validation_split=0.05, callbacks=[tensorboard_cb])

    def compile_model(self):
        # Learning rate?
        self.optimizer = Adam(lr=0.2, beta_1=0.9, beta_2=0.9999,
                                epsilon=1e-08, decay=0.0)
        # Values ofβ2close to 1, required for robust-ness to sparse gradients
        # , Adam performed equal or better than RMSProp, regardless of hyper-parameter setting.

        losses = ['categorical_crossentropy', 'logcosh']  # logcosh to elminate very bad moves, binary win/loss?
        self.model.model.compile(optimizer=self.optimizer, loss=losses)
