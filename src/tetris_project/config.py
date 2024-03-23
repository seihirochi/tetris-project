import numpy as np
from colr import color

from tetris_gym import Action
from tetris_gym.tetris import Mino

EDGE_CHAR = color("　", back="white")
VOID_CHAR = "　"

TETRIS_WIDTH = 10
TETRIS_HEIGHT = 20

mino_I = Mino(
    1,
    np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]
    ]),
    color("　", back="cyan"),
)

mino_J = Mino(
    2,
    np.array([
        [0, 1, 1],
        [0, 1, 0],
        [0, 1, 0]
    ]),
    color("　", back="blue"),
)

mino_L = Mino(
    3,
    np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 1]
    ]),
    color("　", back="orange"),
)

mino_O = Mino(
    4,
    np.array([
        [1, 1],
        [1, 1]
    ]),
    color("　", back="yellow"),
)

mino_S = Mino(
    5,
    np.array([
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1]
    ]),
    color("　", back="green"),
)

mino_T = Mino(
    6,
    np.array([
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 0]
    ]),
    color("　", back="magenta"),
)

mino_Z = Mino(
    7,
    np.array([
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0]
    ]),
    color("　", back="red"),
)

ORDINARY_TETRIS_MINOS = {
    mino_I,
    mino_J,
    mino_L,
    mino_O,
    mino_S,
    mino_T,
    mino_Z,
}

action_LEFT = Action(0, "move left")
action_RIGHT = Action(1, "move right")
action_DOWN = Action(2, "move down")
action_ROTATE_LEFT = Action(3, "rotate left")
action_ROTATE_RIGHT = Action(4, "rotate right")
action_HOLD = Action(5, "hold")
action_HARD_DROP = Action(6, "hard drop")

ORDINARY_TETRIS_ACTIONS = {
    action_LEFT,
    action_RIGHT,
    action_DOWN,
    action_ROTATE_LEFT,
    action_ROTATE_RIGHT,
    action_HOLD,
    action_HARD_DROP,
}

HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP = {
    "a": action_LEFT,
    "d": action_RIGHT,
    "s": action_DOWN,
    "x": action_ROTATE_RIGHT,
    "z": action_ROTATE_LEFT,
    "q": action_HOLD,
    "w": action_HARD_DROP,
}

# action_mode = 1 用の action
# y: -2 ~ width-1, rotate: 0 ~ 3, hold: True/False
ALL_HARDDROP_ACTIONS = set()

for i in range(11): 
    for j in range(4):
        action = Action(i * 4 + j, f"hard drop, move to {i}, rotate {j} times")
        ALL_HARDDROP_ACTIONS.add(action)
ALL_HARDDROP_ACTIONS.add(Action(44, "hold"))
