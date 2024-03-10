import numpy as np
from enum import Enum


class Tetrimino(Enum):
    MINO_I = 1
    MINO_J = 2
    MINO_L = 3
    MINO_O = 4
    MINO_S = 5
    MINO_T = 6
    MINO_Z = 7


MINO_CHAR_MAP = {
    Tetrimino.MINO_I: "I",
    Tetrimino.MINO_J: "J",
    Tetrimino.MINO_L: "L",
    Tetrimino.MINO_O: "O",
    Tetrimino.MINO_S: "S",
    Tetrimino.MINO_T: "T",
    Tetrimino.MINO_Z: "Z",
}

MINO_SHAPE_MAP = {
    # I I I I
    Tetrimino.MINO_I: np.array([[1, 1, 1, 1]]),
    # J . .
    # J J J
    Tetrimino.MINO_J: np.array([[1, 0, 0], [1, 1, 1]]),
    # . . L
    # L L L
    Tetrimino.MINO_L: np.array([[0, 0, 1], [1, 1, 1]]),
    # O O
    # O O
    Tetrimino.MINO_O: np.array([[1, 1], [1, 1]]),
    # . S S
    # S S .
    Tetrimino.MINO_S: np.array([[0, 1, 1], [1, 1, 0]]),
    # . T .
    # T T T
    Tetrimino.MINO_T: np.array([[0, 1, 0], [1, 1, 1]]),
    # Z Z .
    # . Z Z
    Tetrimino.MINO_Z: np.array([[1, 1, 0], [0, 1, 1]]),
}


def to_tetrimino(value: int | float) -> Tetrimino:
    return Tetrimino(int(value))


def get_char_from_board_value(value: int | float) -> str:
    if value == 0:
        return " "
    return MINO_CHAR_MAP[to_tetrimino(value)]


EDGE_CHAR = "#"


class TetrisBoard:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width))

    def set_value(self, x, y, value):
        self.board[y][x] = value

    def set_tetrimino(self, tetrimino, x, y):
        shape = MINO_SHAPE_MAP[tetrimino]
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i][j] == 1:
                    self.set_value(x + j, y + i, tetrimino.value)

    def render_to_string(self):
        s = ""
        s += EDGE_CHAR * (self.width + 2) + "\n"
        for i in range(self.height):
            s += EDGE_CHAR
            for j in range(self.width):
                s += get_char_from_board_value(self.board[i][j])
            s += EDGE_CHAR + "\n"
        s += EDGE_CHAR * (self.width + 2) + "\n"
        return s
