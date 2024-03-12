import numpy as np

from .mino import Mino

EDGE_CHAR = "#"
VOID_CHAR = " "


class TetrisBoard:
    def __init__(self, height: int, width: int, minos: set[Mino]) -> None:
        # === validation ===
        mino_ids = [mino.id for mino in minos]
        if len(mino_ids) != len(set(mino_ids)):
            raise ValueError("Mino id should be unique")

        self.height = height
        self.width = width
        self.board = np.zeros((height, width))
        self.minos = minos
        self.mino_id_map = {mino.id: mino for mino in minos}

    def set_mino_id(self, pos: tuple, mino_id: int) -> None:
        if pos[0] < 0 or pos[0] >= self.height or pos[1] < 0 or pos[1] >= self.width:
            raise ValueError(f"Invalid position: {pos}")
        if mino_id not in self.mino_id_map:
            raise ValueError(f"Invalid mino_id: {mino_id}")

        self.board[pos] = mino_id

    def set_mino(self, mino_id: str, pos: tuple) -> None:
        if pos[0] < 0 or pos[0] >= self.height or pos[1] < 0 or pos[1] >= self.width:
            raise ValueError(f"Invalid position: {pos}")
        if mino_id not in self.mino_id_map:
            raise ValueError(f"Invalid mino_id: {mino_id}")

        mino = self.mino_id_map[mino_id]
        mino_shape = mino.shape
        for i in range(mino_shape.shape[0]):
            for j in range(mino_shape.shape[1]):
                if mino_shape[i][j] == 1:
                    self.set_mino_id((pos[0] + i, pos[1] + j), mino.id)

    def render(self) -> str:
        s = ""
        s += EDGE_CHAR * (self.width + 2) + "\n"
        for i in range(self.height):
            s += EDGE_CHAR
            for j in range(self.width):
                if self.board[i][j] in self.mino_id_map:
                    s += self.mino_id_map[self.board[i][j]].char
                else:
                    s += VOID_CHAR
            s += EDGE_CHAR + "\n"
        s += EDGE_CHAR * (self.width + 2) + "\n"
        return s
