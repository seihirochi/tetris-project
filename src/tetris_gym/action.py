class Action:
    def __init__(self, id: int, name="") -> None:
        self.id = id
        self.name = name

    # action_mode = 1 用の id -> (y, rotate, hold) 変換 method
    def convert_to_tuple(self, width: int) -> tuple:
        hold = (self.id == ((width+1) * 4))
        if hold:
            return 0, 0, True
        y = (self.id % (width+1)) - 2
        rotate = self.id // (width+1)
        return y, rotate, False
