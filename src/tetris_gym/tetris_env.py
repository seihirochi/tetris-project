import random

import gymnasium as gym
import numpy as np

from tetris_gym.tetris import NEXT_MINO_NUM, Mino, Tetris

from .action import Action


class TetrisEnv(gym.Env):
    """
    Tetris の環境を定義
    Env Mode:
    - features: boardを特徴量で表現する
    - natural: boardをそのまま2次元配列で表現する
    """

    def __init__(
        self, minos: set[Mino], action_mode=0, height=20, width=10, env_mode="features"
    ):
        self.view = None
        self.tetris = None
        self.height = height
        self.width = width
        self.minos = minos
        self.action_mode = action_mode
        self.env_mode = env_mode

        if env_mode == "features":
            self.observation_space = gym.spaces.MultiDiscrete(
                [999999999] * 9  # board の特徴量 (上限が不明)
                + [len(minos) + 1]  # current mino
                + [len(minos) + 1] * NEXT_MINO_NUM  # next minos
                + [len(minos) + 1]  # hold mino
            )
        elif env_mode == "natural":
            self.observation_space = gym.spaces.Tuple(
                (
                    gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=(height, width),
                        dtype=np.int32,
                    ),
                    gym.spaces.Discrete(len(minos)),  # current mino
                    gym.spaces.MultiDiscrete(
                        [len(minos)] * NEXT_MINO_NUM
                    ),  # next minos
                    gym.spaces.Discrete(len(minos)),  # hold mino
                )
            )

        if action_mode == 0:
            # Nothing, Left, Right, Rotate left, Rotate right, Drop, Full Drop, Hold
            self.action_space = gym.spaces.Discrete(8)
        elif action_mode == 1:
            self.action_space = gym.spaces.Tuple(
                (
                    gym.spaces.Discrete(width),  # Y (-1 ~ width-1)
                    gym.spaces.Discrete(5),  # Rotation (0 ~ 3: rotate, 4: hold)
                )
            )

    def reset(self, seed=None, options=None) -> tuple:
        # ゲームを初期化 -> tuple( 観測空間, その他の情報 )
        self.tetris = Tetris(self.height, self.width, self.minos)
        # obs = self.tetris.observe()
        obs = (
            self.tetris.observe_natural()
            if self.env_mode == "natural"
            else self.tetris.observe_features()
        )
        info = {}  # other_info
        return obs, info

    def step(self, action: Action) -> tuple:
        # 行動の処理をここで定義
        prev_score = self.tetris.score
        movement_flug = False

        if self.action_mode == 0:
            if action.id == 0:  # move left
                movement_flug = self.tetris.current_mino_state.move(
                    0, -1, self.tetris.board.board
                )
            elif action.id == 1:  # move right
                movement_flug = self.tetris.current_mino_state.move(
                    0, 1, self.tetris.board.board
                )
            elif action.id == 2:  # move down
                if self.tetris.current_mino_state.move(1, 0, self.tetris.board.board):
                    movement_flug = self.tetris.place()
            elif action.id == 3:  # rotate left
                movement_flug = self.tetris.current_mino_state.rotate_left(
                    self.tetris.board.board
                )
            elif action.id == 4:  # rotate right
                movement_flug = self.tetris.current_mino_state.rotate_right(
                    self.tetris.board.board
                )
            elif action.id == 5:  # hold
                movement_flug = self.tetris.hold()
            elif action.id == 6:  # hard drop
                hard_drop_flug = True
                while hard_drop_flug:
                    hard_drop_flug = self.tetris.current_mino_state.move(
                        1, 0, self.tetris.board.board
                    )
                movement_flug = self.tetris.place()
        elif self.action_mode == 1:
            y, rotate, hold_flag = action.convert_to_tuple(self.tetris.board.width)
            if hold_flag:
                movement_flug = self.tetris.hold()
            else:
                movement_flug = self.tetris.move_and_rotate_and_drop(y, rotate)

        if not movement_flug:
            print(f"Movement Failed: {action.name}")

        # このターンで得た報酬
        reward = self.tetris.score - prev_score
        if self.tetris.game_over:
            reward = -1

        if self.env_mode == "natural":
            return (
                self.tetris.observe_natural(),
                reward,
                self.tetris.game_over,
                False,
                {},
            )

        # 設置場所が半分より上か下か
        mino_bottom_x = (
            self.tetris.pre_mino_state.origin[0]
            + self.tetris.pre_mino_state.mino.shape.shape[0]
        )
        else_info = {"is_lower": (self.tetris.board.height // 2) <= mino_bottom_x}

        # tuple(観測情報, 報酬, ゲーム終了フラグ, その他)
        return (
            self.tetris.observe_features(),
            reward,
            self.tetris.game_over,
            False,
            else_info,
        )

    def render(self) -> str:
        return self.tetris.render()

    def seed(self, seed=None):  # Set the random seed for the game
        random.seed(seed)
        return [seed]
