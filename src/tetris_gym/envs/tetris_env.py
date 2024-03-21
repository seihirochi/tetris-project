import random
from typing import Union

import gymnasium as gym
import numpy as np

from tetris_gym import NEXT_MINO_NUM, Action, Mino, Tetris


class TetrisEnv(gym.Env):
    def __init__(self, minos: set[Mino], action_mode=0, height=20, width=10):
        self.view = None
        self.tetris = None
        self.height = height
        self.width = width
        self.minos = minos
        self.action_mode = action_mode

        self.observation_space = gym.spaces.MultiDiscrete(
            [self.height * self.width] * 9 +   # board の特徴量 (Dellacherie's Algorithm)
            [len(minos)+1] +                   # current mino
            [len(minos)+1] * NEXT_MINO_NUM     # next minos
            # [len(minos)+1] +                   # hold mino
        )

        if action_mode == 0:
            # Nothing, Left, Right, Rotate left, Rotate right, Drop, Full Drop, Hold
            self.action_space = gym.spaces.Discrete(8)
        elif action_mode == 1:
            self.action_space = gym.spaces.Tuple((
                gym.spaces.Discrete(width), # Y (-1 ~ width-1)
                gym.spaces.Discrete(5),     # Rotation (0 ~ 3: rotate, 4: hold)
            ))
            
    def get_possible_states(self):
        return self.tetris.get_possible_states()

    def reset(self, seed=None, options=None) -> tuple:
        # ゲームを初期化 -> tuple( 観測空間, その他の情報 )
        self.tetris = Tetris(self.height, self.width, self.minos, self.action_mode)
        obs = self.tetris.observe()
        info = {}  # other_info
        return np.array(obs), info

    def step(self, action: Action) -> tuple:
        prev_score = self.tetris.score
        if self.action_mode == 0:
            if action.id == 0:  # move left
                self.tetris.current_mino_state.move(0, -1, self.tetris.board.board)
            elif action.id == 1:  # move right
                self.tetris.current_mino_state.move(0, 1, self.tetris.board.board)
            elif action.id == 2:  # move down
                flug = self.tetris.current_mino_state.move(1, 0, self.tetris.board.board)
                if flug:
                    self.tetris.place()
            elif action.id == 3:  # rotate left
                self.tetris.current_mino_state.rotate_left(self.tetris.board.board)
            elif action.id == 4:  # rotate right
                self.tetris.current_mino_state.rotate_right(self.tetris.board.board)
            elif action.id == 5:  # hold
                self.tetris.hold()
            elif action.id == 6:  # hard drop
                flug = True
                while flug:
                    flug = self.tetris.current_mino_state.move(1, 0, self.tetris.board.board)
                self.tetris.place()
        elif self.action_mode == 1:
            y, rotate, hold_flag = action.convert_to_tuple(self.tetris.board.width)
            print(y, rotate, hold_flag)
            if hold_flag:
                self.tetris.hold()
            else:
                self.tetris.move_and_rotate_and_drop(y, rotate)

        # このターンで得た報酬
        reward = self.tetris.score - prev_score
        if self.tetris.game_over:
            reward = -1

        # tuple(観測情報, 報酬, ゲーム終了フラグ, {可能な行動集合} )
        return np.array(self.tetris.observe()), reward, self.tetris.game_over, False, {}

    def render(self) -> str:
        return self.tetris.render()
    
    def seed(self, seed=None): # Set the random seed for the game
        random.seed(seed)
        return [seed]
