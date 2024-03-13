import random
from collections import deque

from .board import TetrisBoard
from .mino import Mino
from .mino_state import MinoState

EDGE_CHAR = "\033[48;5;255m　\x1b[0m"
VOID_CHAR = "　"

WALL_WIDTH = 1
NEXT_MINO_NUM = 3
NEXT_MINO_LIST_WIDTH = 6

class Tetris:
    def __init__(self, height: int, width: int, minos: set[Mino]) -> None:
        self.board = TetrisBoard(height, width, minos)
        self.all_mino = minos
        self.mino_permutation = deque()

        # 順列をランダムに shuffle して保持
        add_permutation = list(self.all_mino)
        random.shuffle(add_permutation)
        for mino in add_permutation:
            self.mino_permutation.append(mino)

        # 初期状態でミノを生成
        self.current_mino_state = self._generate_mino_state()
        self.game_over = False

    def _generate_mino_state(self) -> MinoState:
        selected_mino = self.mino_permutation.popleft()

        # len(permutation) < 7 で新しい permutation を puh_back
        if len(self.mino_permutation) < 7:
            add_permutation = list(self.all_mino)
            random.shuffle(add_permutation)
            for mino in add_permutation:
                self.mino_permutation.append(mino)
        
        return MinoState(
            mino=selected_mino,
            height=self.board.height,
            width=self.board.width,
            origin=(0, self.board.width // 2 - selected_mino.shape.shape[1] // 2),
        )

    def is_mino_landed(self) -> bool:
        # mino_state を下にずらして origin が変わるか否かで判定
        # state をコピーして使う
        mino_state = MinoState(
            mino=self.current_mino_state.mino,
            height=self.board.height,
            width=self.board.width,
            origin=self.current_mino_state.origin,
        )
        mino_state.move(1, 0, self.board.board)
        return mino_state.origin == self.current_mino_state.origin

    def step(self) -> None:
        command = input()
        if command == "a":
            self.current_mino_state.move(0, -1, self.board.board)
        elif command == "s":
            # ミノが既に着地していたらボードに固定
            if self.is_mino_landed():
                self.board.set_mino(self.current_mino_state)
                # ラインが揃ったら消す
                self.board.clear_lines()
                # 新しいミノを生成
                self.current_mino_state = self._generate_mino_state()
                # ゲームオーバー判定
                for i in range(self.current_mino_state.mino.shape.shape[0]):
                    for j in range(self.current_mino_state.mino.shape.shape[1]):
                        if self.current_mino_state.mino.shape[i][j] == 1 and self.board.board[self.current_mino_state.origin[0] + i][self.current_mino_state.origin[1] + j] != 0:
                            self.game_over = True
            else:
                self.current_mino_state.move(1, 0, self.board.board)
        elif command == "d":
            self.current_mino_state.move(0, 1, self.board.board)
        elif command == "z":
            self.current_mino_state.rotate_left(self.board.board)
        elif command == "x":
            self.current_mino_state.rotate_right(self.board.board)

    def render(self) -> str:
        all_fields = []
        s = EDGE_CHAR * (self.board.width + 2*WALL_WIDTH)
        all_fields.append(s)

        for i in range(self.board.height):
            s = EDGE_CHAR
            for j in range(self.board.width):
                mino_x = i - self.current_mino_state.origin[0]
                mino_y = j - self.current_mino_state.origin[1]

                if self.board.board[i][j] in self.board.mino_id_map:
                    s += self.board.mino_id_map[self.board.board[i][j]].char
                elif 0 <= mino_x < self.current_mino_state.mino.shape.shape[0] and 0 <= mino_y < self.current_mino_state.mino.shape.shape[1] and self.current_mino_state.mino.shape[mino_x][mino_y] == 1:
                    s += self.current_mino_state.mino.char
                else:
                    s += VOID_CHAR
            s += EDGE_CHAR
            all_fields.append(s)
            
        s = EDGE_CHAR * (self.board.width + 2*WALL_WIDTH)
        all_fields.append(s)

        # Next mino 描画 (4個まで)
        all_fields[0] += VOID_CHAR + "Ｎｅｘｔ" + VOID_CHAR
        now_line = 1
        for i in range(NEXT_MINO_NUM):
            all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
            now_line += 1 # 空行

            for j in range(self.mino_permutation[i].shape.shape[0]):
                s = VOID_CHAR
                if self.mino_permutation[i].id == 4:
                    s += VOID_CHAR # O shape の場合は空白追加

                for k in range(self.mino_permutation[i].shape.shape[1]):
                    if self.mino_permutation[i].shape[j][k] == 1:
                        s += self.mino_permutation[i].char
                    else:
                        s += VOID_CHAR
                s += VOID_CHAR
                all_fields[now_line] += s
                now_line += 1

        # 残りの行を埋める
        while now_line < self.board.height:
            all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
            now_line += 1
            
        s = ""
        for field in all_fields:
            s += field + "\n"
        return s