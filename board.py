###
#
# Benjamin Diamond
# https://github.com/benediamond/chess-alpha-zero
#
###


import chess
import numpy as np
import collections

from chess import Board, STARTING_FEN


class MyBoard(Board):

    def __init__(self, fen=STARTING_FEN):
        Board.__init__(self, fen)

    def __str__(self):
        return self.unicode()

    one_hot = {}
    one_hot[None] = [0, 0, 0, 0, 0, 0]
    one_hot.update(dict.fromkeys(['P', 'p'], [1, 0, 0, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['N', 'n'], [0, 1, 0, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['B', 'b'], [0, 0, 1, 0, 0, 0]))
    one_hot.update(dict.fromkeys(['R', 'r'], [0, 0, 0, 1, 0, 0]))
    one_hot.update(dict.fromkeys(['Q', 'q'], [0, 0, 0, 0, 1, 0]))
    one_hot.update(dict.fromkeys(['K', 'k'], [0, 0, 0, 0, 0, 1]))

    @classmethod
    def _one_hot(cls, piece, side):
        key = piece.symbol() if piece is not None and piece.color == side else None
        return cls.one_hot[key]

    def transposition_key(self):
        return self._transposition_key()

    def num_pieces(self):
        return len(chess.SquareSet(self.occupied_co[chess.WHITE] | self.occupied_co[chess.BLACK]))

    def push_fen(self, fen):
        new = chess.Board(fen)
        if self.fullmove_number == new.fullmove_number and self.turn == chess.WHITE \
                and new.turn == chess.BLACK or self.fullmove_number < new.fullmove_number:
            old_ss = chess.SquareSet(self.occupied_co[self.turn])
            new_ss = chess.SquareSet(new.occupied_co[self.turn])
            diff = list(new_ss.difference(old_ss))
            if len(diff) == 1:
                reverse = list(old_ss.difference(new_ss))
                move = chess.Move(reverse[0], diff[0])
                if self.piece_at(reverse[0]).piece_type != new.piece_at(diff[0]).piece_type:
                    move.promotion = new.piece_at(diff[0]).piece_type
            elif len(diff) == 2:  # castling
                move = chess.Move(self.king(self.turn), new.king(self.turn))
            else:
                raise RuntimeError("problems with pushed fen.")
            self.push(move)
        else:
            self.set_fen(fen)

    def repetitions_count(self):
        transposition_key = self._transposition_key()
        transpositions = collections.Counter()
        transpositions.update((transposition_key, ))

        switchyard = collections.deque()
        while self.move_stack:
            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            transpositions.update((self._transposition_key(), ))

        while switchyard:
            self.push(switchyard.pop())

        return transpositions[transposition_key]

    def gather_features(self, t_history):
        stack = []

        stack.append(np.full((1, 8, 8), self.halfmove_clock))  # np.int64's will later be coerced into np.float64's.
        stack.append(np.full((1, 8, 8), self.has_queenside_castling_rights(chess.BLACK), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.has_kingside_castling_rights(chess.BLACK), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.has_queenside_castling_rights(chess.WHITE), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.has_kingside_castling_rights(chess.WHITE), dtype=np.float64))
        stack.append(np.full((1, 8, 8), self.fullmove_number))
        stack.append(np.full((1, 8, 8), self.turn, dtype=np.float64))
        self._recursive_append(stack, t_history - 1, self.turn)
        return np.concatenate(stack)

    def _recursive_append(self, stack, depth, side):
        if depth > 0:
            move, fen = chess.Move.null(), None
            if self.move_stack:  # there are still moves to pop.
                move = self.pop()
            elif self.is_valid():  # no more moves left, but still valid. we'll clear the board now
                fen = self.fen()
                self.clear()
            self._recursive_append(stack, depth - 1, side)
            if fen:
                self.set_fen(fen)
            else:
                self.push(move)

        repetitions = self.repetitions_count()
        stack.append(np.ones((1, 8, 8)) if repetitions >= 2 else np.zeros((1, 8, 8)))
        stack.append(np.ones((1, 8, 8)) if repetitions >= 3 else np.zeros((1, 8, 8)))

        board_enemy = [self._one_hot(self.piece_at(square), not side) for square in chess.SquareSet(chess.BB_ALL)]
        board_enemy = np.transpose(np.reshape(board_enemy, (8, 8, 6)), (2, 0, 1))
        stack.append(np.flip(board_enemy, 1) if side == chess.WHITE else np.flip(board_enemy, 2))
        board_own = [self._one_hot(self.piece_at(square), side) for square in chess.SquareSet(chess.BB_ALL)]
        board_own = np.transpose(np.reshape(board_own, (8, 8, 6)), (2, 0, 1))
        stack.append(np.flip(board_own, 1) if side == chess.WHITE else np.flip(board_own, 2))
