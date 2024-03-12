import numpy as np

from tetris_gym import Tetris

from .config import ordinary_tetris_minos


def start():
    game = Tetris(20, 10, ordinary_tetris_minos)
    while game.game_over is False:
        game.step()
        print(game.render())
    