import numpy as np


class Mino:
    def __init__(self, id: int, shape: np.array, char: str = "▪︎"):
        # === validation ===
        if id <= 0:
            raise ValueError("id must be positive")
        if shape.shape[0] != shape.shape[1]:
            raise ValueError("Shape must be square")
        if char == "":
            raise ValueError("Char must not be empty")

        self.id = id
        self.shape = shape
        self.char = char

    def __repr__(self) -> str:
        return f"Mino(shape={self.shape}, char={self.char})"

    def __str__(self) -> str:
        return self.char

    def __eq__(self, other) -> bool:
        return self.shape == other.shape and self.char == other.char

    def __hash__(self) -> int:
        return hash(self.id)


class MinoState:
    def __init__(self, mino: Mino, pos: tuple, height: int, width: int):
        self.mino = mino
        self.pos = pos
        self.height = height
        self.width = width

    def __repr__(self) -> str:
        return f"MinoState(mino={self.mino}, pos={self.pos})"

    def rotate_right(self) -> None:
        # 右回転
        prev_shape = self.mino.shape
        self.mino.shape = np.rot90(prev_shape)
        # 場外なら回転前に戻す
        if self.out_field():
            self.mino.shape = prev_shape

    def rotate_left(self) -> None:
        # 左回転
        prev_shape = self.mino.shape
        self.mino.shape = np.rot90(prev_shape, -1)
        # 場外なら回転前に戻す
        if self.out_field():
            self.mino.shape = prev_shape

    def move(self, dx: int, dy: int) -> None:
        prev_pos = self.pos
        # 移動
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        # 場外なら移動前に戻す
        if self.out_field():
            self.pos = prev_pos

    def out_field(self) -> bool:
        # 場外判定
        for i in range(self.mino.shape.shape[0]):
            for j in range(self.mino.shape.shape[1]):
                if self.mino.shape[i][j] == 0:
                    continue
                if (
                    self.pos[0] + i < 0
                    or self.pos[0] + i >= self.height
                    or self.pos[1] + j < 0
                    or self.pos[1] + j >= self.width
                ):
                    return True
        return False
