import random
from typing import Union

import gymnasium as gym
import numpy as np

from tetris_gym import NEXT_MINO_NUM, Mino, Tetris


class TetrisEnv(gym.Env):
    def __init__(self, minos: set[Mino], action_mode=0, height=20, width=10):
        self.view = None
        self.tetris = None
        self.height = height
        self.width = width
        self.minos = minos
        self.action_mode = action_mode

        # 各部分の観測空間を定義
        self.observation_space = gym.spaces.MultiDiscrete(
            [2] * height*width +               # board
            [len(minos)+1, height, width, 4] + # current mino
            [len(minos)+1, height, width, 4] + # hold mino
            [len(minos)+1] * NEXT_MINO_NUM     # next minos
        )
        if action_mode == 0:
            # Nothing, Left, Right, Rotate left, Rotate right, Drop, Full Drop, Hold
            self.action_space = gym.spaces.Discrete(8)
        elif action_mode == 1:
            self.action_space = gym.spaces.Tuple((
                gym.spaces.Discrete(width), # Y
                gym.spaces.Discrete(4),     # Rotation
            ))

    def reset(self, seed=None, options=None) -> tuple:
        self.tetris = Tetris(self.height, self.width, self.minos, self.action_mode)
        obs = self.tetris.observe()
        info = {}  # other_info
        return obs, info

    # int or tuple の action を受け取り、ゲームを進める
    def step(self, action: Union[int, tuple]) -> tuple:
        if self.action_mode == 0:
            if action == 0:  # move left
                self.tetris.current_mino_state.move(0, -1, self.tetris.board.board)
            elif action == 1:  # move right
                self.tetris.current_mino_state.move(0, 1, self.tetris.board.board)
            elif action == 2:  # move down
                prev_origin = self.tetris.current_mino_state.origin
                self.tetris.current_mino_state.move(1, 0, self.tetris.board.board)
                if self.tetris.current_mino_state.origin == prev_origin:
                    self.tetris.place()
            elif action == 3:  # rotate left
                self.tetris.current_mino_state.rotate_left(self.tetris.board.board)
            elif action == 4:  # rotate right
                self.tetris.current_mino_state.rotate_right(self.tetris.board.board)
            elif action == 5:  # hold
                self.tetris.hold()
            elif action == 6:  # hard drop
                prev_origin = None
                while self.tetris.current_mino_state.origin != prev_origin:
                    prev_origin = self.tetris.current_mino_state.origin
                    self.tetris.current_mino_state.move(1, 0, self.tetris.board.board)
                self.tetris.place()
        elif self.action_mode == 1:
            x, y = action
            self._move_and_rotate_and_drop(x, y)

        # tuple(観測情報, 報酬, ゲーム終了フラグ, 追加情報)
        return self.tetris.observe(), self.tetris.score, self.tetris.game_over, False, {}

    def render(self) -> str:
        return self.tetris.render()
    
    def seed(self, seed=None): # Set the random seed for the game
        random.seed(seed)
        return [seed]