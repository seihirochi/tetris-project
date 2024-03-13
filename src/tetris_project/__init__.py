import numpy as np

from tetris_gym import Tetris

from .config import ordinary_tetris_minos


def overwrite_print(text, line):
    print("\033[{};0H\033[K{}".format(line + 1, text))

def start():
    game = Tetris(20, 10, ordinary_tetris_minos)
    while game.game_over is False:
        overwrite_print(game.render(), 0)
        # print(game.render())
        game.step()
    
    # Game Over
    overwrite_print(game.render(), 0)
    # print(game.render())
    