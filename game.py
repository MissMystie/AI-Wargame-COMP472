from __future__ import annotations
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, Iterable
import random
import requests

from coordinates import Coord, CoordPair
from output import Output
from player import Player, Unit, UnitType

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

UNIT_HEURISTIC = [
        9999,  # AI
        3,  # Tech
        3,  # Virus
        3,  # Program
        3,  # Firewall
    ]

##############################################################################################################


@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None
    heuristic_attacker: str | None = "e0"
    heuristic_defender: str | None = "e0"

##############################################################################################################


@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################


@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

    LOG_INVALID_MOVES: bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, player: Player, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        source = self.get(coords.src)
        if source is None or source.player != player:
            return False

        if coords.src != coords.dst:
            adjacent = Coord(coords.src.row, coords.src.col)
            """This checks if the coords are adjacent"""
            adj_checker = False
            for adjacent in adjacent.iter_adjacent():
                if adjacent == coords.dst:
                    adj_checker = True
            if adj_checker is not True:
                return False

        target = self.get(coords.dst)

        if target is None:
            if self.is_restricted_movement(coords.src):
                return False #"invalid move, engaged in battle"
            # makes a extra check for if the unit can move in that direction
            if self.valid_direction(coords):
                return False #f"invalid move, {self.get(coords.src).type} can't move this way"
        elif target is not source and target.player is player:
            if target.health == target.MAX_HEALTH:
                return False #"invalid move, target is at full health"
            elif source.repair_amount(target) <= 0:
                return False #"invalid move, repair amount is 0"

        return True

    def perform_move(self, player: Player, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        #if not self.is_valid_move(player, coords):
            #return False, "invalid move"

        source = self.get(coords.src)
        target = self.get(coords.dst)

        # if moving to an empty slot, return Movement
        if target is None:
            self.set(coords.dst, self.get(coords.src))
            self.set(coords.src, None)
            return True, "moved from " + coords.src.to_string() + " to " + coords.dst.to_string()
        # if target is source, self destruct
        elif target is source:
            self.self_destruct(coords.dst)
            return True, "self-destruct at " + coords.src.to_string()
        # if moving to a slot with an ally unit, repair
        elif target.player is player:
            source.repair_amount(target)
            self.repair(coords)
            return True, coords.src.to_string() + " repaired " + coords.dst.to_string()
        # if moving to a slot with an enemy unit, attack
        elif target.player is not player:
            self.attack(coords)
            return True, coords.src.to_string() + " attacked " + coords.dst.to_string()

        return False, "invalid move"

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self, player: Player, output: Output):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    if self.is_valid_move(player, mv):
                        (success, result) = self.perform_move(player, mv)
                    else:
                        (success, result) = (False, "invalid move")

                    print(f"Broker {player.name}: ", end='')
                    print(result)
                    output.print(player.name + ": " + result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                if self.is_valid_move(player, mv):
                    (success, result) = self.perform_move(player, mv)
                else:
                    (success, result) = (False, "invalid move")

                output.print(player.name + ": " + result)
                if success:
                    print(f"Player {player.name}: ", end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self, player: Player, output: Output) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move(player)
        if mv is not None:
            if self.is_valid_move(player, mv):
                (success, result) = self.perform_move(player, mv)
            else:
                (success, result) = (False, "invalid move")

            output.print(player.name + ": " + result)
            if success:
                print(f"Computer {player.name}: ", end='')
                print(result)
                self.next_turn()
        return mv

    def get_units(self) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None:
                yield coord, unit

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield coord, unit

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self, player: Player) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src, _) in self.player_units(player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(player, move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self, player: Player) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        # (score, move, avg_depth) = self.random_move()
        # (score, move, avg_depth) = self.minimax(self.clone(self))
        (score, move) = minimax(self, player)
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    def is_in_battle(self, coord: Coord):
        """NEW: Checks if the unit is in combat"""
        tmp = Coord(coord.row, coord.col)
        for tmp in tmp.iter_adjacent():
            unit1 = self.get(coord)
            unit2 = self.get(tmp)
            if self.is_valid_coord(tmp) is not None:
                if unit2 is not None:
                    if unit1.player != unit2.player:
                        return True
        return False

    def is_restricted_movement(self, coord: Coord):
        """NEW: Checks if the unit is able to move"""
        if self.is_in_battle(coord) is True:
            if self.get(coord).type is not UnitType.Tech and self.get(coord).type is not UnitType.Virus:
                return True
        return False

    def valid_direction(self, coords: CoordPair):
        """NEW: checks if the unit type is allowed to go in that direction"""
        # checks unit types are AI, Firewall or Program
        if self.get(coords.src).type is not UnitType.Tech and self.get(coords.src).type is not UnitType.Virus:
            # checks for and ignores self destruction cases
            if self.get(coords.src) == self.get(coords.dst):
                return True
            # checks if it's an attacker and calculates accordingly
            if self.get(coords.src).player == Player.Attacker:
                if coords.dst.col == (coords.src.col - 1) or coords.dst.row == (coords.src.row - 1):
                    return False
            # checks if it's a defender and calculates accordingly
            if self.get(coords.src).player == Player.Defender:
                if coords.dst.col == (coords.src.col + 1) or coords.dst.row == (coords.src.row + 1):
                    return False
        return True

    def attack(self, coords: CoordPair) -> bool:
        """NEW: hurts the target at the specified coordinates"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            print("invalid coordinates!")
            return False

        source = self.get(coords.src)
        target = self.get(coords.dst)

        self.mod_health(coords.dst, -source.damage_amount(target))
        self.mod_health(coords.src, -target.damage_amount(source))

        self.remove_dead(coords.src)
        self.remove_dead(coords.dst)

        return True

    def repair(self, coords: CoordPair) -> bool:
        """NEW: repairs the target at the specified coordinates"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        source = self.get(coords.src)
        target = self.get(coords.dst)

        if target.is_full_health():
            return False

        self.mod_health(coords.dst, source.repair_amount(target))

        return True

    def self_destruct(self, coord: Coord):
        """NEW: self destructs hurting all units around it"""
        tmp = Coord(coord.row, coord.col)
        for tmp in tmp.iter_range(1):
            if self.is_valid_coord(tmp):
                if self.is_empty(tmp) is not None:
                    self.mod_health(tmp, -2)
                    self.remove_dead(tmp)
        self.mod_health(coord, -10)
        self.remove_dead(coord)

    def print_settings(self, output: Output):
        output.print("Current settings:\n")
        output.print("Timeout in seconds: " + str(self.options.max_time) + "\n")
        output.print("Max number of Turns: " + str(self.options.max_turns) + "\n")
        output.print("Alpha Beta is on: " + str(self.options.alpha_beta) + "\n")
        output.print("Game Type: " + str(self.options.game_type) + "\n")

        if self.options.game_type == GameType.CompVsComp:
            output.print("Attacker Heuristic: " + self.options.heuristic_attacker + "\n")
            output.print("Defender Heuristic: " + self.options.heuristic_defender + "\n")
        if self.options.game_type == GameType.AttackerVsComp:
            output.print("Defender Heuristic: " + self.options.heuristic_defender + "\n")
        if self.options.game_type == GameType.CompVsDefender:
            output.print("Attacker Heuristic: " + self.options.heuristic_attacker + "\n")


##############################################################################################################


class Node:
    game_state: Game = None
    player: Player = Player.Attacker
    move: CoordPair
    parent: Node = None
    successors: list[Node | None]
    depth: int = 0
    score = 0
    alpha = MIN_HEURISTIC_SCORE
    beta = MAX_HEURISTIC_SCORE
    alphaBeta: bool = True

    def __init__(self, game_state: Game, player: Player, move: CoordPair, parent: Node, depth: int, score: int = 0, alpha: int = alpha, beta: int = beta, alphaBeta: bool = True):
        self.game_state = game_state
        self.player = player
        self.move = move
        self.parent = parent
        self.depth = depth
        self.score = score
        self.alpha = alpha
        self.beta = beta
        self.alphaBeta = alphaBeta
        self.successors = []
        return

    def get_successors(self) -> list[Node]:
        next_player = self.player.next()
        for move in list(self.game_state.move_candidates(self.player)):
            new_game_state = self.game_state.clone()
            new_game_state.perform_move(self.player, move)
            successor = Node(new_game_state, next_player, move, self, self.depth + 1)
            self.successors.append(successor)

        return self.successors

    # returns true hits the depth limit
    def at_max_depth(self):
        return self.depth >= self.game_state.options.max_depth

    # returns true if no children, returns false if it has children
    def is_terminal(self, max_depth: int):
        return len(self.successors) == 0


##############################################################################################################


def minimax(current_game_state: Game, current_player: Player) -> (int, CoordPair):
    # TODO max or min based on defender or attacker

    #checking if AlphaBeta is true
    alphaBeta = current_game_state.options.alpha_beta
    root = Node(current_game_state, current_game_state.next_player, None, None, 0, alphaBeta)

    # grabs the best value from leaf nodes
    max_value(root, current_player)
    # grabs the children of the node to check for the best move
    for successor in root.successors:
        print(successor.score)
        if successor.score == root.score:
            best_move = successor.move
            #break

    # score, move, avg_depth
    return root.score, best_move


def max_value(node: Node, current_player: Player) -> Node:
    if node.at_max_depth() or len(node.get_successors()) == 0:
        node.score = get_utility(node, current_player)
        return node
    if node.parent is not None and node.parent.alphaBeta == True:
        node.alphaBeta = node.parent.alphaBeta #if alphabeta is on or off
    node.alpha = MIN_HEURISTIC_SCORE #alpha

    for s in node.successors:
        node.alpha = max(node.alpha, min_value(s, current_player).score)
        #TODO implement alpha beta here, break loop if alphabeta (use parent node)
        if node.parent is not None and node.alpha >= node.parent.beta and node.alphaBeta == True: #if enabled & alpha > beta
            break

    node.score = node.alpha
    #print("Player: " + node.player.name + ", Max Score: " + str(node.score))

    # propagates values up the tree
    return node


def min_value(node: Node, current_player: Player) -> Node:
    if node.at_max_depth() or len(node.get_successors()) == 0:
        node.score = get_utility(node, current_player)
        return node

    if node.parent is not None and node.parent.alphaBeta == True:
        node.alphaBeta = node.parent.alphaBeta #if alphabeta is on or off
    node.beta = MAX_HEURISTIC_SCORE #beta

    for s in node.successors:
        node.beta = min(node.beta, max_value(s, current_player).score) # beta
        # TODO implement alpha beta here, break loop if alphabeta (use parent node) 
        if node.parent is not None and node.parent.alpha >= node.beta and node.alphaBeta == True: #if enabled & alpha > beta
            return node #immediately propagate up the tree

    node.score = node.beta
    #print("Player: " + node.player.name + ", Min Score: " + str(node.score))

    # propagates values up the tree
    return node


def get_utility(node: Node, current_player: Player) -> int:
    assert node is not None

    #TODO different heuristics based on attacker/defender
    if current_player == Player.Attacker:
        heuristic = node.game_state.options.heuristic_attacker
    else:
        heuristic = node.game_state.options.heuristic_defender

    match heuristic:
        case "e0":
            utility = heuristic_e0(node.game_state, current_player)
        case "e1":
            utility = heuristic_e1(node.game_state, current_player)
        case "e2":
            utility = heuristic_e2(node.game_state, current_player)
        case _:
            utility = heuristic_e0(node.game_state, current_player)

    #print("\tMove: " + node.move.to_string() + ", Depth: " + str(node.depth) + ", Score: " + str(utility))
    return utility


# grabs the heuristic score from the proper heuristic
def heuristic_e0(game_state: Game, current_player: Player) -> int:
    heuristic_score = 0

    for u in game_state.get_units():
        coord = u[0]
        unit = u[1]
        if unit.player is current_player:
            heuristic_score += UNIT_HEURISTIC[unit.type.value]
        else:
            heuristic_score -= UNIT_HEURISTIC[unit.type.value]

    return heuristic_score


# grabs the heuristic score from the proper heuristic
def heuristic_e1(game_state: Game, current_player: Player) -> int:
    heuristic_score = 0

    for u in game_state.get_units():
        coord = u[0]
        unit = u[1]
        if unit.player is current_player:
            heuristic_score += round(((unit.MAX_HEALTH / 2) + ((unit.health / unit.MAX_HEALTH) / 2)) * UNIT_HEURISTIC[unit.type.value])
        else:
            heuristic_score -= round(((unit.MAX_HEALTH / 2) + ((unit.health / unit.MAX_HEALTH) / 2)) * UNIT_HEURISTIC[unit.type.value])

    return heuristic_score

def heuristic_e2(game_state: Game, current_player: Player) -> int:
    heuristic_score = 0

    for u in game_state.get_units():
        coord = u[0]
        unit = u[1]
        if unit.player is current_player:
            heuristic_score += round(((unit.MAX_HEALTH / 2) + ((unit.health / unit.MAX_HEALTH) / 2)) * UNIT_HEURISTIC[unit.type.value])
        else:
            heuristic_score -= round(((unit.MAX_HEALTH / 2) + ((unit.health / unit.MAX_HEALTH) / 2)) * UNIT_HEURISTIC[unit.type.value])

    return heuristic_score