#!/usr/bin/python3
"""

RUN.PY
Parses arguments from terminal.


"""


import os
import sys
import logging
import argparse
import multiprocessing as mp
from dotenv import load_dotenv, find_dotenv

from config import Config
from logging import StreamHandler, Formatter

if find_dotenv():
    load_dotenv(find_dotenv())

CMD_LIST = ["tui", "self", "alpha"]
CONFIG_LIST = ["normal", "test", "home", "delivery"]

log = logging.getLogger(__name__)


# Make sure our files can be reached
path = os.path.dirname(os.path.dirname(__file__))
if path not in sys.path:
    sys.path.append(path)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="What do you want to run?", choices=CMD_LIST, default="tui")
    parser.add_argument("--config", help="Config type", choices=CONFIG_LIST, default="normal")
    args = parser.parse_args()

    # Config
    config_type = args.config
    config = Config(config_type=config_type)
    config.resource.create_directories()

    # Logger
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=config.resource.main_log_path, level=logging.DEBUG, format=log_format)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(log_format))
    logging.getLogger().addHandler(stream_handler)
    log.info("Config type: %s", config_type)

    # To avoid "RecursionError: maximum recursion depth exceeded while pickling an object" when multiprocessing,
    # but you might get the "OverflowError: Maximum recursion level reached" as well, idk
    mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)

    if args.cmd == "tui":
        from tui import tui
        tui.start(config)
    if args.cmd == "self":
        import self_play
        self_play.start(config)
    if args.cmd == "delivery":
        import self_play
        self_play.start(config)
    if args.cmd == "alpha":
        import pseudo
        pseudo.start(config)
