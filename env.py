"""

ENV


"""

import enum
import copy
import chess
import numpy
import collections

from chess import Board
from chess import STARTING_FEN

from config import Config

# Settings->Editor->inspections->Python->Incorrect Call Arguments
Result = enum.Enum('Result', 'WHITE BLACK DRAW')


class ChessEnv:
    def __init__(self, config: Config):
        self.board = MyBoard()
        self.config = config
        self.result = None
        self.resigned = False

    def reset(self):
        self.board = MyBoard()
        self.result = None
        self.resigned = False
        return self

    def step(self, move):
        if move == chess.Move.null() or move is None:
            self.resign()
            return

        self.board.push(move)

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

    def render(self):
        print(self.board)
        print("\n")

    def resign(self):  # Do we even resign?
        self.resigned = True
        print("RESIGNING")
        if self.board.turn == chess.WHITE:
            self.result = Result.BLACK
        else:
            self.result = Result.WHITE

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    @property
    def done(self):
        return self.result is not None

    def fen(self):
        return self.board.fen()

    def pos_key(self):
        return self.board.transposition_key()

    def get_result(self):
        if self.result == Result.WHITE:
            return 1
        elif self.result == Result.BLACK:
            return -1
        elif self.result == Result.DRAW:
            return 0
        else:
            return None

    @property
    def next_to_move(self):
        if self.board.turn == chess.BLACK:
            return 1
        else:
            return -1


class MyBoard(Board):
    def __init__(self, fen=STARTING_FEN):
        Board.__init__(self, fen)
        self.one_hot = {}
        self.fill_one_hot()

    def fill_one_hot(self):
        self.one_hot[None] = [0, 0, 0, 0, 0, 0]
        self.one_hot.update(dict.fromkeys(["P", "p"], [1, 0, 0, 0, 0, 0]))
        self.one_hot.update(dict.fromkeys(["N", "n"], [0, 1, 0, 0, 0, 0]))
        self.one_hot.update(dict.fromkeys(["B", "b"], [0, 0, 1, 0, 0, 0]))
        self.one_hot.update(dict.fromkeys(["R", "r"], [0, 0, 0, 1, 0, 0]))
        self.one_hot.update(dict.fromkeys(["Q", "q"], [0, 0, 0, 0, 1, 0]))
        self.one_hot.update(dict.fromkeys(["K", "k"], [0, 0, 0, 0, 0, 1]))

    def transposition_key(self):
        return self._transposition_key()

    def repetition_count(self):
        """Taken from python chess _can_claim_threefold_repetition_ (needed as one of the inputs to NN)"""
        key = self._transposition_key()
        transpositions = collections.Counter()
        transpositions.update((key, ))

        switchyard = collections.deque()
        while self.move_stack:
            move = self.pop()
            switchyard.append(move)
            if self.is_irreversible(move):
                break
            transpositions.update((self._transposition_key(), ))

        while switchyard:
            self.push(switchyard.pop())

        return transpositions[key]

    def _one_hot(self, piece, side):
        if piece is not None and side:
            key = piece.symbol()
        else:
            key = None
        return self.one_hot[key]

    def make_image(self, depth):  # depth: Number of half-moves back into the game history
        """
        The input to the neural network is an N x N (MT + L) image stack that represents state
        using a concatenation of T sets of M planes of size N x N.
        The board is oriented to the perspective of the current player.
        M indicates the presence of the player's pieces, with one plane for each piece type,
        and a second set of planes for the opponent's pieces.
        L denotes the player's colour, the move number, legality of castling, repetition count,
        and number of moves without progress.
        """
        # M
        features = []  # Settings->Editor->inspections->Python->List
        switchyard = collections.deque()
        clear = 0  # number of clear boards (when movestack runs out)

        for _ in range(depth):
            if self.move_stack:
                move = self.pop()
                switchyard.append(move)
            else:
                clear += 1

        for _ in range(depth):
            if clear > 0:
                clear -= 1
                self.clear()
                player_board = []
                for square in chess.SquareSet(chess.BB_ALL):
                    player_board.append(self._one_hot(self.piece_at(square), True))
                player_board = numpy.transpose(numpy.reshape(player_board, (8, 8, 6)), (2, 0, 1))
                if self.turn == chess.WHITE:
                    features.append(numpy.flip(player_board, 1))
                else:
                    features.append(numpy.flip(player_board, 2))

                opponent_board = []
                for square in chess.SquareSet(chess.BB_ALL):
                    opponent_board.append(self._one_hot(self.piece_at(square), False))
                player_board = numpy.transpose(numpy.reshape(player_board, (8, 8, 6)), (2, 0, 1))
                if self.turn == chess.WHITE:
                    features.append(numpy.flip(player_board, 1))
                else:
                    features.append(numpy.flip(player_board, 2))

                # There is no repetitions if the board is cleared
                features.append(numpy.zeros((1, 8, 8)))
                features.append(numpy.zeros((1, 8, 8)))

                if clear == 0:
                    self.set_fen(STARTING_FEN)
            else:
                self.push(switchyard.pop())
                player_board = []
                for square in chess.SquareSet(chess.BB_ALL):
                    player_board.append(self._one_hot(self.piece_at(square), True))
                player_board = numpy.transpose(numpy.reshape(player_board, (8, 8, 6)), (2, 0, 1))
                if self.turn == chess.WHITE:
                    features.append(numpy.flip(player_board, 1))
                else:
                    features.append(numpy.flip(player_board, 2))

                opponent_board = []
                for square in chess.SquareSet(chess.BB_ALL):
                    opponent_board.append(self._one_hot(self.piece_at(square), False))
                player_board = numpy.transpose(numpy.reshape(player_board, (8, 8, 6)), (2, 0, 1))
                if self.turn == chess.WHITE:
                    features.append(numpy.flip(player_board, 1))
                else:
                    features.append(numpy.flip(player_board, 2))

                repetitions = self.repetition_count()
                if repetitions >= 2:
                    features.append(numpy.ones((1, 8, 8)))
                else:
                    features.append(numpy.zeros((1, 8, 8)))
                if repetitions >= 3:
                    features.append(numpy.ones((1, 8, 8)))
                else:
                    features.append(numpy.zeros((1, 8, 8)))

        # L
        features.append(numpy.full((1, 8, 8), self.turn, dtype=numpy.float64))  # Is float necessary though?
        features.append(numpy.full((1, 8, 8), self.fullmove_number, dtype=numpy.float64))
        features.append(numpy.full((1, 8, 8), self.has_kingside_castling_rights(chess.WHITE), dtype=numpy.float64))
        features.append(numpy.full((1, 8, 8), self.has_queenside_castling_rights(chess.WHITE), dtype=numpy.float64))
        features.append(numpy.full((1, 8, 8), self.has_kingside_castling_rights(chess.BLACK), dtype=numpy.float64))
        features.append(numpy.full((1, 8, 8), self.has_queenside_castling_rights(chess.BLACK), dtype=numpy.float64))
        features.append(numpy.full((1, 8, 8), self.halfmove_clock, dtype=numpy.float64))  # might have to make this int

        # debug
        # print(numpy.concatenate(features))
        return numpy.concatenate(features)
