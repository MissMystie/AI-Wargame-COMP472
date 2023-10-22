from __future__ import annotations
from dataclasses import dataclass

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
from coordinates import CoordPair
from game import Game

MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


@dataclass(slots=True)
class MiniMax:

    @staticmethod
    def minimax(self, current_game_state: Game) -> (int, CoordPair):

        #TODO max or min based on defender or attacker

        root = Node(current_game_state, None, None, 0)
        # grabs the best value from leaf nodes
        self.max_value(root)
        # grabs the children of the node to check for the best move

        for successor in root.successors:
            if successor.score == root.score:
                best_move = successor.move
                break

        # score, move, avg_depth
        return root.score, best_move

    @staticmethod
    def max_value(self, node: Node) -> Node:

        if node.atMaxDepth() or len(node.getSuccessors()) == 0:
            node.score = self.getUtility(node)
            return node

        max_value = MIN_HEURISTIC_SCORE

        for node in node.successors:
            max_value = max(max_value, self.min_value(node).score)

        node.score = max_value

        # propogates values up the tree
        return node

    @staticmethod
    def min_value(self, node: Node) -> Node:

        if node.atMaxDepth() or len(node.getSuccessors()) == 0:
            node.score = self.getUtility(node)
            return node

        min_value = MAX_HEURISTIC_SCORE

        for node in node.successors:
            min_value = min(min_value, self.max_value(node).score)

        node.score = min_value

        # propogates values up the tree
        return node

    @staticmethod
    def getUtility(self, node: Node):
        assert node is not None
        return self.getHeuristicScore(node.game_state)

    @staticmethod
    # grabs the heuristic score from the proper heuristic
    def getHeuristicScore(self, game_state: Game) -> int:

        heuristicScore = 0

        return heuristicScore


class Node:
    game_state: Game = None
    move: CoordPair
    parent: Node = None
    successors: list[Node | None]
    depth: int = 0
    score = 0

    def __init__(self, game_state: Game, move: CoordPair, parent: Node, depth: int, score: int = 0):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.currentNode = None
        self.depth = 0
        self.successors = []
        return

    def getSuccessors(self) -> list[Node]:
        for move in list(self.game_state.move_candidates()):
            new_game_state = self.game_state.clone
            new_game_state.perform_move(move)
            successor = Node(new_game_state, move, self, self.depth + 1)
            self.successors.append(successor)

        return self.successors

    # returns true hits the depth limit
    def atMaxDepth(self):
        return self.depth >= self.game_state.options.max_depth

    # returns true if no children, returns false if it has children
    def isTerminal(self, max_depth: int):
        return len(self.successors) == 0

