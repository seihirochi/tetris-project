import numpy as np

from tetris_gym import Mino, TetrisBoard

from .config import mino_I, mino_T, ordinary_tetris_minos


def start():
    board = TetrisBoard(20, 10, ordinary_tetris_minos)
    board.set_mino(mino_I.id, (3, 3))
    board.set_mino(mino_T.id, (6, 5))

    print(board.render())
