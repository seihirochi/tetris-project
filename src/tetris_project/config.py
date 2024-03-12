# from enum import Enum

# import numpy as np


# class Tetrimino(Enum):
#     MINO_I = 1
#     MINO_J = 2
#     MINO_L = 3
#     MINO_O = 4
#     MINO_S = 5
#     MINO_T = 6
#     MINO_Z = 7


# MINO_CHAR_MAP = {
#     Tetrimino.MINO_I: "I",
#     Tetrimino.MINO_J: "J",
#     Tetrimino.MINO_L: "L",
#     Tetrimino.MINO_O: "O",
#     Tetrimino.MINO_S: "S",
#     Tetrimino.MINO_T: "T",
#     Tetrimino.MINO_Z: "Z",
# }

# MINO_SHAPE_MAP = {
#     # . I . .
#     # . I . .
#     # . I . .
#     # . I . .
#     Tetrimino.MINO_I: np.array(
#         [
#             [0, 1, 0, 0],
#             [0, 1, 0, 0],
#             [0, 1, 0, 0],
#             [0, 1, 0, 0],
#         ]
#     ),
#     # . J J
#     # . J .
#     # . J .
#     Tetrimino.MINO_J: np.array(
#         [
#             [0, 1, 1],
#             [0, 1, 0],
#             [0, 1, 0],
#         ]
#     ),
#     # . L .
#     # . L .
#     # . L L
#     Tetrimino.MINO_L: np.array(
#         [
#             [0, 1, 0],
#             [0, 1, 0],
#             [0, 1, 1],
#         ]
#     ),
#     # O O
#     # O O
#     Tetrimino.MINO_O: np.array(
#         [
#             [1, 1],
#             [1, 1],
#         ]
#     ),
#     # . S .
#     # . S S
#     # . . S
#     Tetrimino.MINO_S: np.array(
#         [
#             [0, 1, 0],
#             [0, 1, 1],
#             [0, 0, 1],
#         ]
#     ),
#     # . T .
#     # . T T
#     # . T .
#     Tetrimino.MINO_T: np.array(
#         [
#             [0, 1, 0],
#             [0, 1, 1],
#             [0, 1, 0],
#         ]
#     ),
#     # . . Z
#     # . Z Z
#     # . Z .
#     Tetrimino.MINO_Z: np.array(
#         [
#             [0, 0, 1],
#             [0, 1, 1],
#             [0, 1, 0],
#         ]
#     ),
# }

import numpy as np

from tetris_gym import Mino

mino_I = Mino(
    1,
    np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),
    "Ｉ",
)

mino_J = Mino(
    2,
    np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0]]),
    "ｊ",
)

mino_L = Mino(
    3,
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]]),
    "Ｌ",
)

mino_O = Mino(
    4,
    np.array([[1, 1], [1, 1]]),
    "Ｏ",
)

mino_S = Mino(
    5,
    np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]]),
    "Ｓ",
)

mino_T = Mino(
    6,
    np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]]),
    "Ｔ",
)

mino_Z = Mino(
    7,
    np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0]]),
    "Ｚ",
)


ordinary_tetris_minos = {
    mino_I,
    mino_J,
    mino_L,
    mino_O,
    mino_S,
    mino_T,
    mino_Z,
}
