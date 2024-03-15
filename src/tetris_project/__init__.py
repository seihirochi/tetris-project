from tetris_gym import Tetris

from .config import (
    ORDINARY_TETRIS_MINOS,
    ORDINARY_TETRIS_ACTIONS,
    HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
)
from .controller import HumanController


def overwrite_print(text, line):
    print("\033[{};0H\033[K{}".format(line + 1, text))


def start():
    game = Tetris(20, 10, ORDINARY_TETRIS_MINOS, ORDINARY_TETRIS_ACTIONS)
    controller = HumanController(
        ORDINARY_TETRIS_ACTIONS,
        HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
    )
    while game.game_over is False:
        overwrite_print(game.render(), 0)
        action = controller.get_action()
        game.step(action.id)

    # Game Over
    overwrite_print(game.render(), 0)
