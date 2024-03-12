import numpy as np

from tetris_gym import Tetris

from .config import ordinary_tetris_minos
import time
import pynput

TICK = 0.05


def start():
    game = Tetris(20, 10, ordinary_tetris_minos)
    action = None
    TICK_COUNT = 0

    # pynputを使ってactionに値を入れる
    def on_press(key):
        nonlocal action
        if key == pynput.keyboard.Key.left:
            action = "a"
        elif key == pynput.keyboard.Key.right:
            action = "d"
        elif key == pynput.keyboard.Key.down:
            action = "s"
        elif key == pynput.keyboard.Key.space:
            action = "x"
        elif key == pynput.keyboard.Key.esc:
            action = "z"

    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()

    while game.game_over is False:
        TICK_COUNT += 1
        time.sleep(TICK)
        if TICK_COUNT % 20 == 0:
            game.step("s")
        elif action is not None:
            game.step(action)
        action = None
        print(game.render() )
         