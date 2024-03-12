import numpy as np

from tetris_gym import Tetris

from .config import mino_I, mino_T, ordinary_tetris_minos
# import keyboard
import time

TICK = 0.1


def start():
    game = Tetris(20, 10, ordinary_tetris_minos)
    while True:
        game.step(0)
        print(game.render())
        time.sleep(TICK)
