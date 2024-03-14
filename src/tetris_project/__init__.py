import numpy as np

from tetris_gym import Tetris

from .config import ordinary_tetris_minos

ACTION_MAP = {
    "a": 0,  # 左に移動
    "d": 1,  # 右に移動
    "s": 2,  # 下に移動
    "z": 3,  # 左に回転
    "x": 4,  # 右に回転
    "q": 5,  # ホールド
    "w": 6   # ハードドロップ
}

def overwrite_print(text, line):
    print("\033[{};0H\033[K{}".format(line + 1, text))

def start():
    game = Tetris(20, 10, ordinary_tetris_minos)
    while game.game_over is False:
        overwrite_print(game.render(), 0)
        # print(game.render())
        command = input()
        if command in ACTION_MAP:
            command = ACTION_MAP[command]
        game.step(command)
    
    # Game Over
    overwrite_print(game.render(), 0)
