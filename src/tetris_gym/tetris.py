import random

from .board import TetrisBoard
from .mino import Mino
from .mino_state import MinoState

EDGE_CHAR = "＃"
VOID_CHAR = "　"


class Tetris:
    def __init__(self, height: int, width: int, minos: set[Mino]) -> None:
        self.board = TetrisBoard(height, width, minos)
        # 初期状態でミノを生成
        self.current_mino_state = self._generate_mino_state()
        self.game_over = False
        # 初期状態の render
        print(self.render())

    def _generate_mino_state(self) -> MinoState:
        selected_mino = random.choice(list(self.board.minos))
        return MinoState(
            mino=selected_mino,
            height=self.board.height,
            width=self.board.width,
            origin=(0, self.board.width // 2 - selected_mino.shape.shape[1] // 2),
        )

    def is_mino_landed(self) -> bool:
        # mino_state を下にずらして origin が変わるか否かで判定
        # state をコピーして使う
        mino_state = MinoState(
            mino=self.current_mino_state.mino,
            height=self.board.height,
            width=self.board.width,
            origin=self.current_mino_state.origin,
        )
        mino_state.move(1, 0, self.board.board)

        return mino_state.origin == self.current_mino_state.origin

    def step(self, action: str) -> None:
        if action == "s":
            self.current_mino_state.move(1, 0, self.board.board)
        elif action == "a":
            self.current_mino_state.move(0, -1, self.board.board)
        elif action == "d":
            self.current_mino_state.move(0, 1, self.board.board)
        elif action == "z":
            self.current_mino_state.rotate_left(self.board.board)
        elif action == "x":
            self.current_mino_state.rotate_right(self.board.board)

        # ミノが着地したらボードに固定
        if self.is_mino_landed():
            self.board.set_mino(self.current_mino_state)
            # 新しいミノを生成
            self.current_mino_state = self._generate_mino_state()
            # ゲームオーバー判定
            for i in range(self.current_mino_state.mino.shape.shape[0]):
                for j in range(self.current_mino_state.mino.shape.shape[1]):
                    if (
                        self.current_mino_state.mino.shape[i][j] == 1
                        and self.board.board[self.current_mino_state.origin[0] + i][
                            self.current_mino_state.origin[1] + j
                        ]
                        != 0
                    ):
                        self.game_over = True
                        return

    def render(self) -> str:
        s = ""
        s += EDGE_CHAR * (self.board.width + 2) + "\n"
        for i in range(self.board.height):
            s += EDGE_CHAR
            for j in range(self.board.width):
                mino_x = i - self.current_mino_state.origin[0]
                mino_y = j - self.current_mino_state.origin[1]

                if self.board.board[i][j] in self.board.mino_id_map:
                    s += self.board.mino_id_map[self.board.board[i][j]].char
                elif (
                    0 <= mino_x < self.current_mino_state.mino.shape.shape[0]
                    and 0 <= mino_y < self.current_mino_state.mino.shape.shape[1]
                    and self.current_mino_state.mino.shape[mino_x][mino_y] == 1
                ):
                    s += self.current_mino_state.mino.char
                else:
                    s += VOID_CHAR
            s += EDGE_CHAR + "\n"
        s += EDGE_CHAR * (self.board.width + 2) + "\n"
        return s
