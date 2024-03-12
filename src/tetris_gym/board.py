import numpy as np

from .mino import Mino
from .mino_state import MinoState


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

    def set_mino(self, state: MinoState) -> None:
        for i in range(state.mino.shape.shape[0]):
            for j in range(state.mino.shape.shape[1]):
                if state.mino.shape[i][j] == 1:
                    self.set_mino_id((state.origin[0] + i, state.origin[1] + j), state.mino.id)

