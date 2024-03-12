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
