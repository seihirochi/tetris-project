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
