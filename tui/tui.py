"""

Chess Simulation in terminal
-PLay against AI


"""

import time
import chess

from env import ChessEnv
from config import Config
from tui.logic import PlayAI


def start(config: Config):
    start_time = time.time()

    chess_model = PlayAI(config)
    env = ChessEnv(config)
    human_is_white = True
    chess_model.start(human_is_white)

    moves = 0
    while not env.done():
        moves += 1
        if (env.board.turn == chess.WHITE) == human_is_white:
            action = chess_model.move_by_ai(env)
            print("Your move: " + str(action))
        else:
            action = chess_model.move_by_ai(env)
            print("AI move: " + str(action) + "\n")
        env.step(action)
        env.render()

    end_time = time.time()
    print("Game ended in " + str(end_time - start_time) + " seconds")
    print("Moves: " + str(moves))
    print("Result: " + env.board.result())
