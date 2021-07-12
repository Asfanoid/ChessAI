"""

PLAYER


"""
import chess
import chess.pgn
import random
import numpy as np
from time import time

from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from env import ChessEnv, Result
from config import Config


class StateAction:
    def __init__(self):
        self.a = defaultdict(ActionDetails)  # (move, details)
        self.sum_of_n = 0               # Sum of visit count
        self.p = 0                  # Probability


class ActionDetails:
    def __init__(self):
        self.n = 0  # Visit count
        self.w = 0  # Total action-value
        self.q = 0  # Mean action-Value
        self.p = 0  # Probability of selecting Action (a) in State (s)


class ChessPlayer:
    def __init__(self, config: Config, pipes):
        self.config = config
        self.labels = config.labels
        self.n_labels = config.n_labels
        self.tree = defaultdict(StateAction)
        self.data = []
        self.pipes = pipes
        self.node_lock = defaultdict(Lock)

    def reset_search_tree(self):
        self.tree = defaultdict(StateAction)

    def action(self, env, random_move=False):
        self.reset_search_tree()

        if random_move:  # Just return a random move (for testing)
            move_list = list(env.board.generate_legal_moves())
            rand = random.randrange(len(move_list))
            return move_list[rand]

        threads = self.config.num_threads
        simulations = self.config.num_simulations
        with ThreadPoolExecutor(max_workers=threads) as executor:
            values = executor.map(self.search, [env.copy() for _ in range(simulations)])

        # for _ in range(simulations):
            # values = self.search(env.copy())
        target_policy, policy = self.calculate_policy(env)
        action = int(np.random.choice(range(self.n_labels), p=policy))

        for move in self.tree[env.pos_key()].a.keys():
            if self.labels[move] == action:
                self.data.append([str(move), list(target_policy)])
                #print("move:", str(move))
                return move

    def search(self, env):
        """ Summary of Alpha Zero Search, see paper (supplementary data) at page 15.
        # Store action-state pair with N, W, Q, and P
        # Simulation begins at root-node of search tree, finishes when reaching leaf node (not rollout!)
        # At each leaf node, an action is selected (a=argmax(Q+U), using PUCT algorithm (U=CPsqr(N)/(1+N))
        # Leaf node goes through neural network evaluation (P, v = f) (to get probability and value)
        # Leaf node is expanded and stateActionPair is initialized (N=0, W=0, Q=0, P=Pa)
        # Then update visit count (N=N+1), Value (W=W+v), and Mean value (Q=W/N) in a backward manner
        """
        if env.done:  # No need to continue if the side to move just lost
            if env.result == Result.DRAW:
                return 0
            else:
                return -1  # return leaf value

        key = env.pos_key()

        # Expansion
        with self.node_lock[key]:
            if key not in self.tree:
                leaf_probability, leaf_value = self.calculate_p_and_v(env)
                self.tree[key].p = leaf_probability  # Store policy (probability for all moves) for later
                return leaf_value

            # Selection
            move = self.selection(env)
            if move is None:
                print("SELECTED NONE MOVE")

            state_details = self.tree[key]
            action_details = state_details.a[move]

            state_details.sum_of_n += 1
            action_details.n += 1
            action_details.w += -1  # Make sure value is less than unvisited ones?

        env.step(move)
        leaf_value = -self.search(env)
        
        with self.node_lock[key]:
            # BackPropagation
            action_details.w += 1 + leaf_value
            action_details.q = action_details.w / action_details.n

        return leaf_value

    def selection(self, env):
        # Add probability to all legal moves
        state_details = self.tree[env.pos_key()]
        if state_details.p is not None:
            counter = 0
            for move in env.board.legal_moves:
                probability = state_details.p[self.labels[move]]
                state_details.a[move].p = probability
                counter += probability
            for details in state_details.a.values():
                details.p /= counter  # Make sure total probability equals 1
            state_details.p = None

        """
        Calculate the best moves based on probability
        a = argmax(Q(s,a) + U(s,a)) where Q is mean value of an action, and U is
        U(s,a) = C(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a)), Paper supplementary data page 15 (Search)
        """

        # Noise Epsilon?
        c = 3  # Exploration rate? NEEDS ADJUSTMENTS
        sr = np.sqrt(state_details.sum_of_n + 1)  # static

        move_value = defaultdict(str)
        for move, details in state_details.a.items():
            move_value[move] = (details.q + (c * details.p * sr / (1 + details.n)))

        # Why was this here again??
        # if len(move_value.values()) == 1:
            # return
        best_move = max(move_value, key=move_value.get)
        #best_move = np.argmax(move_value)
        #print("best_move", str(best_move))
        return best_move

    def calculate_policy(self, env):
        state_details = self.tree[env.pos_key()]
        policy = np.zeros(self.n_labels)

        for move in state_details.a.keys():
            # Fill policy with number of visits
            if move is not None:
                policy[self.labels[move]] = state_details.a[move].n
            else:
                print("MOVE IS NONE")

        target_policy = policy / state_details.sum_of_n

        best_move = np.argmax(policy)
        max_policy = np.zeros(self.n_labels)
        max_policy[best_move] = 1

        if env.board.fullmove_number < self.config.num_sampling_moves:  # Explore more at beginning (30 moves)
            return target_policy, target_policy
        else:  # Only choose the best action
            return target_policy, max_policy

    def calculate_p_and_v(self, env):
        #time_image = time()
        image = env.board.make_image(self.config.t_history)
        #print("image:", time() - time_image)
        #time_pipe = time()
        policy, value = self.predict(image)
        #print("pipe:", time() - time_pipe)

        if env.board.turn == chess.BLACK:
            #time_change_policy = time()
            policy = self.change_pov(policy)
            #print("change_policy", time() - time_change_policy)

        # BEFORE NEURAL NETWORK WORKAROUND
        # leaf_probability = random.random()  # IF POLICY IS ZERO IT MIGHT BREAK
        # leaf_value = random.random()

        return policy, value

    def predict(self, image):
        pipe = self.pipes.pop()
        pipe.send(image)
        ret = pipe.recv()
        self.pipes.append(pipe)
        return ret

    def add_result(self, result):
        for move in self.data:
            move.append(result)

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
