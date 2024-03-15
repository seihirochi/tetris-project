from gymnasium.envs.registration import register

from .action import *
from .board import *
from .mino import *
from .mino_state import *
from .tetris import *

register(
    id='tetris-v1',
    entry_point='tetris_gym.envs:TetrisEnv',
)