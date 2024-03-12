import numpy as np


class Mino:
    def __init__(self, id: int, shape: np.array, fulcrum: tuple, char: str = "▪︎"):
        # === validation ===
        if id <= 0:
            raise ValueError("id must be positive")
        if shape.shape[0] != shape.shape[1]:
            raise ValueError("Shape must be square")
        if len(fulcrum) != 2:
            raise ValueError("Fulcrum must be a tuple of length 2. format: (x, y)")
        if (
            fulcrum[0] < 0
            or fulcrum[0] >= shape.shape[0]
            or fulcrum[1] < 0
            or fulcrum[1] >= shape.shape[1]
        ):
            raise ValueError("Fulcrum must be on the shape")
        if char == "":
            raise ValueError("Char must not be empty")

        self.id = id
        self.shape = shape
        self.char = char
        self.fulcrum = fulcrum

    def __repr__(self) -> str:
        return f"Mino(shape={self.shape}, char={self.char})"

    def __str__(self) -> str:
        return self.char

    def __eq__(self, other) -> bool:
        return self.shape == other.shape and self.char == other.char

    def __hash__(self) -> int:
        return hash(self.id)


class MinoState:
    def __init__(self, mino: Mino, height: int, width: int):
        self.mino = mino
        self.height = height
        self.width = width

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
        # 移動
        prev_flucrum = self.mino.fulcrum
        self.mino.fulcrum = (prev_flucrum[0] + dx, prev_flucrum[1] + dy)
        # 場外なら移動前に戻す
        if self.out_field():
            self.mino.fulcrum = prev_flucrum

    def out_field(self) -> bool:
        # 場外判定
        for i in range(self.shape.shape[0]):
            for j in range(self.shape.shape[1]):
                if self.mino.shape[i][j] == 0:
                    continue
                if (
                    self.fulcrum[0] + i < 0
                    or self.fulcrum[0] + i >= self.height
                    or self.fulcrum[1] + j < 0
                    or self.fulcrum[1] + j >= self.width
                ):
                    return True
        return False
