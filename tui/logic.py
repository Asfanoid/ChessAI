"""

TUI_LOGIC


"""

import chess

from config import Config
from player import ChessPlayer


class PlayAI:
    def __init__(self, config: Config):
        self.human_color = None
        self.ai = None  # initialised later
        self.config = config

    def start(self, human_is_white):
        if human_is_white:
            self.human_color = chess.WHITE
        else:
            self.human_color = chess.BLACK

        self.ai = ChessPlayer(self.config)

    def move_by_human(self, env):
        # Print legal moves as numbered list
        for idx, move in enumerate(env.board.legal_moves):
            print(idx + 1, move)

        while True:
            try:
                move = input("(UCI) Enter Move: ")  # UCI Format?
                if str(move) == "exit":
                    print("EXITING")  # Exit (exit)
                    # sys.exit()
                    return chess.Move.null()  # Same as resign
                move = chess.Move.from_uci(move)
                if env.board.is_legal(move):
                    return move
                else:
                    print("Invalid Move")
            except:
                print("Invalid Move")

    def move_by_ai(self, env):
        return self.ai.action(env)
