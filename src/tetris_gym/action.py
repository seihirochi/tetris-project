class Action:
    def __init__(self, id: int, name: str, y=0, rotate=0) -> None:
        self.id = id
        self.name = name
        # ==== action_mode が 1 の場合のみ使用 ==== #
        self.y = y # y軸方向の変量
        self.rotate = rotate # 回転回数