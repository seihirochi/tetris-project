from gymnasium.envs.registration import register

from .action import *
from .tetris_env import TetrisEnv

register(
    id="tetris-v1",
    entry_point="tetris_gym:TetrisEnv",
)
