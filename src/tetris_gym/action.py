from typing import Tuple


class Action:
    def __init__(self, id: int, name="") -> None:
        self.id = id
        self.name = name

    # action_mode = 1 用の id -> (y, rotate, hold) 変換 method
    def convert_to_tuple(self, width: int) -> Tuple[int, int, bool]:
        hold = self.id == ((width + 1) * 4)
        if hold:
            return 0, 0, True
        y = (self.id % (width + 1)) - 2
        rotate = self.id // (width + 1)
        return y, rotate, False

    def __lt__(self, other) -> bool:
        return self.id < other.id
    
    def __eq__(self, other) -> bool:
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __str__(self) -> str:
        return f"Action(id={self.id}, name={self.name})"
    
    def __repr__(self) -> str:
        return str(self)
