import numpy as np

from config import Config
from helpers import make_network, save_network, load_network

from keras.optimizers import Adam
from keras.callbacks import TensorBoard


class TrainingModel:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
        self.model = make_network(self.config, name="training_")
        self.optimizer = None

    def start(self, dataset):
        # load_network(self.config, self.model, name="training_")
        self.model = make_network(self.config, name="training_")
        self.compile_model()
        self.train_epoch(dataset)
        save_network(self.config, self.model, name="training_")
        # save_network(self.config, self.model, name="model_")

    def train_epoch(self, dataset):
        batch_size = 32  # 2048
        epochs = 1
        state_deque, policy_deque, value_deque = dataset
        state_ary, policy_ary, value_ary = np.asarray(state_deque), np.asarray(policy_deque), np.asarray(value_deque)
        tensorboard_cb = TensorBoard(log_dir=self.config.resource.log_dir, batch_size=batch_size, histogram_freq=1)
        self.model.model.fit(state_ary, [policy_ary, value_ary], batch_size=batch_size, epochs=epochs, shuffle=True,
                             validation_split=0.05, callbacks=[tensorboard_cb])

    def compile_model(self):
        print("Compiling")
        # Learning rate?
        self.optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.model.compile(optimizer=self.optimizer, loss=losses)
