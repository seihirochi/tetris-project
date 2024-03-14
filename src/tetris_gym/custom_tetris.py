from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .mino import Mino
from .tetris import Tetris


class CustomTetrisEnv(gym.Env):
    def __init__(self, height: int, width: int, minos: set[Mino]):
        super().__init__()
        self.tetris = Tetris(height, width, minos)
        self.action_space = spaces.Discrete(7)  # アクション定義
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=255, shape=(height, width), dtype=int),  # テトリスボードの状態
            'mino_origin': spaces.MultiDiscrete([height, width]),  # ミノの位置 (x, y) ※ 負も取り得るので注意
            'mino_shape': spaces.Box(low=0, high=1, shape=(4, 4), dtype=int)  # ミノの形状 (4x4の正方行列)
        })

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.tetris.step(action)
        observation = self.tetris.get_observation()
        reward = self.tetris.get_reward()
        done = self.tetris.game_over
        info = {} # その他の情報
        return observation, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        # 環境をリセットし、初期状態の観測を返す
        self.tetris.reset()
        return self.tetris.get_observation()

    def render(self, mode='human') -> None:
        self.tetris.render()
