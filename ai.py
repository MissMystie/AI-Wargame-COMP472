from __future__ import annotations
from dataclasses import dataclass

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
from game import Game

MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


@dataclass(slots=True)
class MiniMax:

    def __init__(self, game_tree: Game):
        self.game_tree = game_tree
        self.root = game_tree
        self.currentNode = None
        self.successors = []
        self.depth = 0
        return

    def minimax(self, node):

        # grabs the best value of the parent node
        best_val = self.max_value(node)
        # grabs the children of the node to check for the best move
        successors = self.getSuccessors(node)

        # propogates values up the tree
        # score, move, avg_depth
        best_move = None
        for elem in successors:
            if elem.value == best_val:
                best_move = elem
                break
        return best_move

    def max_value(self, node):
        if self.isTerminal(node):
            return self.getUtility(node)

        max_value = MIN_HEURISTIC_SCORE

        successors_states = self.getSuccessors(node)
        for state in successors_states:
            max_value = max(max_value, self.min_value)
        return max_value

    def min_value(self, node):
        if self.isTerminal(node):
            return self.getUtility(node)

        min_value = MAX_HEURISTIC_SCORE

        successors_states = self.getSuccessors(node)
        for state in successors_states:
            min_value = min(min_value, self.max_value)
        return min_value

    def getSuccessors(self, node):
        assert node is not None
        return list(self.move_candidates(node))

    # returns true if no children or hits the depth limit, returns false if it has children
    def isTerminal(self, node):
        assert node is not None
        if node.depth > self.game_tree.max_depth:
            return True
        return len(self.move_candidates(node)) == 0

    def getUtility(self, node):
        assert node is not None
        return self.getHeuristicScore(node)

    def getHeuristicScore(self, node):
        # grabs the heuristic score from the proper heuristic
        heuristicScore = 0

        return heuristicScore
