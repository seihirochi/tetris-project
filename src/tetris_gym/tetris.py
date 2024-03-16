import copy
import random
from collections import deque
from typing import List, Tuple, Union

import numpy as np

from .board import TetrisBoard
from .mino import Mino
from .mino_state import MinoState

EDGE_CHAR = "\033[48;5;255m　\x1b[0m"
VOID_CHAR = "　"

WALL_WIDTH = 1
NEXT_MINO_NUM = 3
NEXT_MINO_LIST_WIDTH = 6
LINE_CLEAR_SCORE = [0, 100, 300, 500, 800]


class Tetris:
    def __init__(
        self, height: int, width: int, minos: set[Mino], action_mode=0
    ) -> None:
        self.board = TetrisBoard(height, width, minos)
        self.mino_permutation = deque()
        self.minos = minos  # 全種類の mino
        self.action_mode = action_mode

        # self.hold_mino = Mino(0, np.array([[0]]), VOID_CHAR)  # hold している mino
        self.hold_mino = MinoState(
            mino=Mino(0, np.array([[0]]), VOID_CHAR),
            height=height,
            width=width,
            origin=(0, 0),
        )
        self.hold_used = False  # 今のターンに hold したか否か

        self.line_total_count = 0
        self.score = 0
        self.turns = 0

        # 初期状態でミノを生成
        self.current_mino_state = self._generate_mino_state()
        self.game_over = False

    def observe(self) -> np.ndarray:
        return np.concatenate(
            [
                self.board.to_tensor().flatten(),
                self.current_mino_state.to_tensor().flatten(),
                self.hold_mino.to_tensor().flatten(),
                # NEXT_MINO_NUM 個までの next mino を 1 次元に変換
                np.concatenate(
                    [mino.to_tensor().flatten() for mino in self.mino_permutation][:NEXT_MINO_NUM]
                )
            ]
        )

    def _generate_mino_state(self) -> MinoState:
        # len(permutation) < 7 で新しい permutation を puh_back
        if len(self.mino_permutation) < 7:
            add_permutation = copy.deepcopy(list(self.minos))
            random.shuffle(add_permutation)
            for mino in add_permutation:
                self.mino_permutation.append(mino)

        self.turns += 1
        selected_mino = self.mino_permutation.popleft()

        return MinoState(
            mino=selected_mino,
            height=self.board.height,
            width=self.board.width,
            origin=(0, self.board.width // 2 - selected_mino.shape.shape[1] // 2),
        )

    def hold(self) -> bool:
        if self.hold_used:
            return False

        self.hold_used = True
        if self.hold_mino.mino.id == 0:
            self.hold_mino = self.current_mino_state
            self.current_mino_state = self._generate_mino_state()
        else: # swap
            self.hold_mino, self.current_mino_state = self.current_mino_state, self.hold_mino
        return True

    def place(self) -> None:
        self.score += 1 # 設置出来たら +1 点
        self.hold_used = False  # hold 状況をリセット
        self.board.set_mino(self.current_mino_state)  # ミノをボードに固定

        line_count = self.board.clear_lines()  # ラインが揃ったら消す
        self.line_total_count += line_count
        self.score += LINE_CLEAR_SCORE[line_count]  # スコア加算

        self.current_mino_state = self._generate_mino_state()  # 新しいミノを生成

        # ゲームオーバー判定
        for i in range(self.current_mino_state.mino.shape.shape[0]):
            for j in range(self.current_mino_state.mino.shape.shape[1]):
                if (
                    self.current_mino_state.mino.shape[i][j] == 1
                    and self.board.board[self.current_mino_state.origin[0] + i][
                        self.current_mino_state.origin[1] + j
                    ]
                    != 0
                ):
                    self.game_over = True
    
    def move_and_rotate_and_drop(self, y: int, rotate: int) -> bool:
        # (y座標変位, 回転回数) -> 移動可能 flag
        prev_state = copy.deepcopy(self.current_mino_state)
        flag = True

        # move
        while y != self.current_mino_state.origin[0]:
            if y > self.current_mino_state.origin[0]:
                flag = self.current_mino_state.move(0, -1, self.board.board)
                y -= 1
            elif y < self.current_mino_state.origin[0]:
                flag = self.current_mino_state.move(0, 1, self.board.board)
                y += 1
            if not flag:
                self.current_mino_state = prev_state
                return False
        # rotate
        while rotate > 0:
            flag = self.current_mino_state.rotate_left(self.board.board)
            rotate -= 1
            if not flag:
                self.current_mino_state = prev_state
                return False
        # drop
        while flag:
            flag = self.current_mino_state.move(1, 0, self.board.board)
        self.place()
        return True
    
    def get_possible_actions(self) -> List[Tuple[Union[int, list], np.ndarray]]:
        # List( Tuple( 可能な行動, その状態 )) を返す
        actions = []
        if self.action_mode == 0:
            # actions = [1, 2, 3, 4, 5, 6, 7]
            # ※ 現在は train として使わないので一旦スルー
            pass
        elif self.action_mode == 1:
            for y in range(self.board.width):
                for rotate in range(4):
                    # 行動出来るかを確認
                    # ※ 本来ここは deepcopy ではなく差分更新で高速化すべき
                    Tetris_copy = copy.deepcopy(self)
                    flag = Tetris_copy.move_and_rotate_and_drop(y, rotate)
                    if flag:
                        actions.append(((y, rotate), Tetris_copy.observe()))
        return actions

    def render(self) -> str:
        all_fields = []
        s = EDGE_CHAR * (self.board.width + 2 * WALL_WIDTH)
        all_fields.append(s)

        for i in range(self.board.height):
            s = EDGE_CHAR
            for j in range(self.board.width):
                mino_x = i - self.current_mino_state.origin[0]
                mino_y = j - self.current_mino_state.origin[1]

                if self.board.board[i][j] in self.board.mino_id_map:
                    s += self.board.mino_id_map[self.board.board[i][j]].char
                elif (
                    0 <= mino_x < self.current_mino_state.mino.shape.shape[0]
                    and 0 <= mino_y < self.current_mino_state.mino.shape.shape[1]
                    and self.current_mino_state.mino.shape[mino_x][mino_y] == 1
                ):
                    s += self.current_mino_state.mino.char
                else:
                    s += VOID_CHAR
            s += EDGE_CHAR
            all_fields.append(s)

        s = EDGE_CHAR * (self.board.width + 2 * WALL_WIDTH)
        all_fields.append(s)

        # Next mino 描画 (4個まで)
        all_fields[0] += VOID_CHAR + "Ｎｅｘｔ" + VOID_CHAR
        now_line = 1
        for i in range(NEXT_MINO_NUM):
            all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
            now_line += 1  # 空行

            for j in range(self.mino_permutation[i].shape.shape[0]):
                s = VOID_CHAR
                if self.mino_permutation[i].id == 4:
                    s += VOID_CHAR  # O shape の場合は空白追加

                for k in range(self.mino_permutation[i].shape.shape[1]):
                    if self.mino_permutation[i].shape[j][k] == 1:
                        s += self.mino_permutation[i].char
                    else:
                        s += VOID_CHAR
                s += VOID_CHAR
                all_fields[now_line] += s
                now_line += 1

        # Next mino 描画 (4個まで)
        all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
        now_line += 1  # 空行
        all_fields[now_line] += VOID_CHAR + "Ｈｏｌｄ" + VOID_CHAR
        now_line += 1
        all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
        now_line += 1  # 空行

        if self.hold_mino is not None:
            for i in range(self.hold_mino.mino.shape.shape[0]):
                s = VOID_CHAR
                if self.hold_mino.mino.id == 4:
                    s += VOID_CHAR
                for j in range(self.hold_mino.mino.shape.shape[1]):
                    if self.hold_mino.mino.shape[i][j] == 1:
                        s += self.hold_mino.mino.char
                    else:
                        s += VOID_CHAR
                s += VOID_CHAR
                all_fields[now_line] += s
                now_line += 1

        # 残りの行を埋める
        while now_line < self.board.height + 2 * WALL_WIDTH:
            all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
            now_line += 1

        # 画面下部にスコアとライン数を表示
        s = VOID_CHAR + "Score " + VOID_CHAR + "Line" + VOID_CHAR * 11
        all_fields.append(s)
        s = (
            VOID_CHAR
            + f"{self.score:0>6}"
            + VOID_CHAR
            + f"{self.line_total_count:0>6}"
            + VOID_CHAR * 10
        )
        all_fields.append(s)

        # 下部を見やすくするようの空行
        s = VOID_CHAR * (self.board.width + 2 * WALL_WIDTH)
        all_fields.append(s)

        s = ""
        for field in all_fields:
            s += field + "\n"
        return s
