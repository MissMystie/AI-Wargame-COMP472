from __future__ import annotations
import argparse

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
from game import Game, GameType, Player, Options
from output import Output


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--max_turns', type=int, help='maximum number of turns')
    parser.add_argument('--heuristic_attacker', type=str, help='minimax heuristic: e0|e1|e2')
    parser.add_argument('--heuristic_defender', type=str, help='minimax heuristic: e0|e1|e2')
    parser.add_argument('--alpha_beta', type=str, help='is alpha-beta pruning enabled')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.heuristic_attacker is not None:
        options.heuristic_attacker = args.heuristic_attacker
    if args.heuristic_defender is not None:
        options.heuristic_defender = args.heuristic_defender
    if args.alpha_beta is not None:
        options.alpha_beta = (args.alpha_beta.lower() == 'true')

    # create a new game
    game = Game(options=options)

    # create Trace file and loaded up with starting info
    output = Output(str(options.alpha_beta), str(options.max_time), str(options.max_turns))
    game.print_settings(output)

    # the main game loop
    while True:
        output.print("\n" + game.to_string())
        winner = game.has_winner()

        if winner is not None:
            output.print(f"{winner.name} wins!")
            break

        player = game.next_player
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn(player, output)
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn(player, output)
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn(player, output)
        else:
            move = game.computer_turn(player, output)
            if move is not None:
                game.post_move_to_broker(move)
            else:
                output.print(f"Computer {player.name} : execution interrupted")
                output.print(f"{player.next().name} wins!")
                break
    output.close()

##############################################################################################################


if __name__ == '__main__':
    main()
