# AI-Wargame-COMP472

## Launching the game
To launch the game run this command

python3 ai_wargame.py

### Game parameters
You can add game parameters by adding them to the end of the command above, the current list of game parameters are as follows:

param, valid input type, description

--max_depth, int, maximum search depth

--max_time, float, maximum search time
    
--game_type, str, game types: auto|attacker|defender|manual (PARTIALLY IMPLEMENTED (D2 will implement the rest))

--broker, str, play via a game broker (BROKER NOT IMPLEMENTED YET (D2 will implement this param))

--max_turns, int, maximum number of turns

## How to play

Inputs are taken in the following format:

Starting Space/Ending Space, Ex: A3A4
### Moving
Each piece can move to an adjacent space with a few exceptions, the movement input is as follows:

Unit space/Empty adjacent space, Ex: B0B1, where B0 is a Unit and B1 is a empty space
### Healing
Certain pieces can heal each other as long as the to be healed unit isn't at full health and is an adjacent teammate, the healing input is as follows:

Unit that's healing/To be healed unit, Ex D1D2, where D1 is the unit who heals and D2 is an ally unit not at full health
### Attacking
Pieces can attack adjacent enemy units, the attacking input is as follows:

Unit that's attacking/Enemy unit, C1B1, where C1 is the attacking unit and B1 is the Enemy unit (Note that both units attack each other during this process)
### Self destructing
All pieces can self destruct, the self destruct input is as follows:

Unit that's self destructing/Unit that's self destructing, Ex B3B3, where B3 is the Unit self destructing

