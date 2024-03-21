class Action:
    def __init__(self, id: int, name="") -> None:
        self.id = id
        self.name = name

    # action_mode = 1 の為の constructor 作成
    @classmethod
    def from_values(cls, y: int, rotate: int, hold: bool, width: int) -> "Action":
        # y: [-1, width-1], rotate: [0, 3], hold: bool
        if hold:
            return cls(width*4, "hold")
        return cls(
            y+1 + rotate*width,
            f"y={y}, rotate={rotate}"
        )
    
    def convert_to_tuple(self, width: int) -> tuple:
        # id -> (y, rotate, hold) # action_mode = 1 対応
        hold = (self.id == (width * 4))
        if hold:
            return 0, 0, True
        y = (self.id % width) - 1
        rotate = self.id // width
        return y, rotate, False
