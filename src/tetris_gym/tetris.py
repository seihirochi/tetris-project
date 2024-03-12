from .board import TetrisBoard
from .mino import Mino, MinoState

import random


class Tetris:
    def __init__(self, height: int, width: int, minos: set[Mino]) -> None:
        self.current_mino_state = None
        self.board = TetrisBoard(height, width, minos)

    def _generate_mino_state(self) -> MinoState:
        selected_mino = random.choice(list(self.board.minos))
        return MinoState(
            mino=selected_mino,
            pos=(0, (self.board.width - selected_mino.shape.shape[1]) // 2),
            height=self.board.height,
            width=self.board.width,
        )

    def is_mino_landed(self) -> bool:
        if self.current_mino_state is None:
            return False
        mino = self.current_mino_state.mino
        pos = self.current_mino_state.pos
        # ミノの下端がフィールドの下端に接しているか
        if pos[0] + mino.shape.shape[0] >= self.board.height:
            return True
        # ミノの下端が他のミノに接しているか
        bottoms = []
        for i in range(mino.shape.shape[1]):
            for j in range(mino.shape.shape[0], 0, -1):
                if mino.shape[j - 1][i] == 1:
                    bottoms.append((j - 1, i))
                    break
        for bottom in bottoms:
            if (
                pos[0] + bottom[0] + 1 < self.board.height
                and self.board.board[pos[0] + bottom[0] + 1][pos[1] + bottom[1]] != 0
            ):
                return True
        return False

    def step(self, action=0) -> None:
        prev_mino_pos = self.current_mino_state.pos if self.current_mino_state else None
        if self.current_mino_state is None:
            self.current_mino_state = self._generate_mino_state()
        else:
            if action == 0:
                self.current_mino_state.move(1, 0)
            elif action == 1:
                self.current_mino_state.move(0, -1)
            elif action == 2:
                self.current_mino_state.move(0, 1)
            elif action == 3:
                self.current_mino_state.rotate(1)
            elif action == 4:
                self.current_mino_state.rotate(-1)
            elif action == 5:
                self.current_mino_state.move(-1, 0)

        # add current mino to board
        if prev_mino_pos is not None:
            self.board.remove_mino(self.current_mino_state.mino.id, prev_mino_pos)
        self.board.set_mino(
            self.current_mino_state.mino.id, self.current_mino_state.pos
        )

        # check if the mino is landed
        if self.is_mino_landed():
            self.current_mino_state = None

    def render(self) -> str:
        return self.board.render()
