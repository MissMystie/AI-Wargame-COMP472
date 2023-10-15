from __future__ import annotations
from dataclasses import dataclass
from typing import TextIO

from game import GameType, Options


@dataclass(slots=True)
class Output:
    OUTPUT_PATH = "output/"
    FILE_EXTENSION = "txt"

    file: TextIO | None

    def __init__(self, options: Options, game_type: GameType):
        b = options.alpha_beta
        t = options.max_time
        m = options.max_turns

        filename = "gameTrace-" + str(b) + "-" + str(t) + "-" + str(m) + "." + self. FILE_EXTENSION
        self.file = open(self.OUTPUT_PATH + filename, "w+")

        self.file.write("Current settings:\n")
        self.file.write("Timeout in seconds: " + str(t) + "\n")
        self.file.write("Max number of Turns: " + str(m) + "\n")
        self.file.write("Alpha Beta is on: " + str(b) + "\n")
        self.file.write("Game Type: " + str(game_type) + "\n")

        self.file.flush()

        return

    def print(self, line: str):
        if line is not None:
            self.file.write(line + "\n")
            self.file.flush()
        return

    def close(self):
        self.file.close
        return