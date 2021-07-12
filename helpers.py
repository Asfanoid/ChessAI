import os

from glob import glob
from config import Config
from network import Network
from datetime import datetime


def load_network(config, network, name="model_"):
    pattern = os.path.join(config.resource.model_dir, name + "*")
    directories = list(sorted(glob(pattern)))
    if not directories:
        print("Not directories")
        return False
    directory = directories[-1]
    path_config = os.path.join(directory, name + "config.json")
    path_weight = os.path.join(directory, name + "weight.h5")
    network.load(path_config, path_weight)
    return True


def alpha_load_network(config, network, name="model_"):
    pattern = os.path.join(config.resource.model_dir, name + "*")
    directories = list(sorted(glob(pattern)))
    if not directories:
        print("Not directories")
        return
    directory = directories[-1]
    print("Loading model:", directory)
    path_config = os.path.join(directory, name + "config.json")
    path_weight = os.path.join(directory, name + "weight.h5")
    network.load(path_config, path_weight)


# Deprecated?
def load_network_after_training(config, network):
    model_pattern = os.path.join(config.resource.model_dir, "model_*")
    model_directories = list(sorted(glob(model_pattern)))
    if not model_directories:
        print("Not model directories")
        return False
    model_directory = model_directories[-1]
    training_pattern = os.path.join(config.resource.model_dir, "training_*")
    training_directories = list(sorted(glob(training_pattern)))
    if not training_directories:
        print("Not training directories")
        return False
    training_directory = training_directories[-1]
    path_config = os.path.join(model_directory, "model_config.json")
    path_weight = os.path.join(training_directory, "training_weight.h5")
    network.load(path_config, path_weight)
    return True


def save_network(config, network, name="model_"):
    save_dir = config.resource.model_dir
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    path_save = os.path.join(save_dir, name + date)
    os.makedirs(path_save, exist_ok=True)
    path_config = os.path.join(path_save, name + "config.json")
    path_weight = os.path.join(path_save, name + "weight.h5")
    return network.save(path_config, path_weight)


def make_network(config: Config, name="model_"):
    network = Network(config)
    if not load_network(config, network, name):
        network.build()
        save_network(config, network, name)
        load_network(config, network, name)
        print("New network")
    return network
