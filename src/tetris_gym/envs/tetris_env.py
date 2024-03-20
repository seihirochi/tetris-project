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

        # Dellacherie's algorithm.
        self.observation_space = gym.spaces.MultiDiscrete(
            # [2] * height*width +               # board
            [self.height * self.width] * 9 + # board の特徴量
            [len(minos)+1] +                   # current mino
            # [len(minos)+1] +                 # hold mino
            [len(minos)+1] * NEXT_MINO_NUM     # next minos
        )

        if action_mode == 0:
            # Nothing, Left, Right, Rotate left, Rotate right, Drop, Full Drop, Hold
            self.action_space = gym.spaces.Discrete(8)
        elif action_mode == 1:
            self.action_space = gym.spaces.Tuple((
                gym.spaces.Discrete(width-1), # Y
                gym.spaces.Discrete(4),     # Rotation
            ))
            
    def get_possible_states(self):
        return self.tetris.get_possible_states()

    def reset(self, seed=None, options=None) -> tuple:
        # ゲームを初期化 -> tuple( 観測空間, その他の情報 )
        self.tetris = Tetris(self.height, self.width, self.minos, self.action_mode)
        obs = self.tetris.observe()
        info = {}  # other_info
        return np.array(obs), info

    def step(self, action: Union[int, tuple]) -> tuple:
        # action_mode = 0 : 0, 1, 2, 3, 4, 5, 6
        # action_mode = 1 : action => tuple(action/width, action%width) => (y, rotate)

        prev_score = self.tetris.score

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
            y, rotate = action
            # print(f"\ny: {y}, rotate: {rotate}")
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
