from __future__ import annotations
from dataclasses import dataclass
from typing import TextIO

@dataclass(slots=True)
class Output:
    OUTPUT_PATH = "output/"
    FILE_EXTENSION = "txt"

    file: TextIO | None


    def __init__(self, alpha_beta: str, max_time: str, max_turns: str):

        filename = "gameTrace-" + alpha_beta + "-" + max_time + "-" + max_turns + "." + self.FILE_EXTENSION
        self.file = open(self.OUTPUT_PATH + filename, "w+")

        return

    def print(self, line: str):
        if line is not None:
            print(line)
            self.file.write(line + "\n")
            self.file.flush()
        return

    def close(self):
        self.file.close