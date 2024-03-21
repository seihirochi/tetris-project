import copy
import random
import numpy as np
from collections import deque

from .board import TetrisBoard
from .mino import Mino
from .action import Action
from .mino_state import MinoState

EDGE_CHAR = "\033[48;5;255m　\x1b[0m"
VOID_CHAR = "　"

WALL_WIDTH = 1
NEXT_MINO_NUM = 3
NEXT_MINO_LIST_WIDTH = 6
LINE_CLEAR_SCORE = [0, 100, 300, 500, 800]


class TetrisState:
    def __init__(
        self,
        board: TetrisBoard,
        mino_state: MinoState,
        hold_mino: Mino,
        next_minos: list[Mino],
        score: int,
        line_total_count: int,
        turn: int,
    ) -> None:
        self.board = board
        self.mino_state = mino_state
        self.hold_mino = hold_mino
        self.next_minos = next_minos
        self.score = score
        self.line_total_count = line_total_count
        self.turn = turn
        self.size = self.get_size()

    def to_tensor(self) -> np.ndarray:
        return np.concatenate(
            [
                self.board.to_tensor().flatten(),
                self.mino_state.to_tensor().flatten(),
                np.array([self.mino_state.origin[0], self.mino_state.origin[1]]),
                self.hold_mino.to_tensor().flatten(),
                np.concatenate(
                    [mino.to_tensor().flatten() for mino in self.next_minos]
                ),
                np.array([self.score, self.line_total_count, self.turn]),
            ]
        )

    def get_size(self) -> int:
        return sum(
            [
                self.board.to_tensor().flatten().size,
                self.mino_state.to_tensor().flatten().size,
                2,
                self.hold_mino.to_tensor().flatten().size,
                sum([mino.to_tensor().flatten().size for mino in self.next_minos]),
                3,
            ]
        )


class Tetris:
    def __init__(
        self, height: int, width: int, minos: set[Mino], actions: set[Action]
    ) -> None:
        self.board = TetrisBoard(height, width, minos)
        self.mino_permutation = deque()
        self.minos = minos  # 全種類の mino
        self.action_map = {action.id: action for action in actions}

        self.hold_mino = Mino(0, np.array([[0]]), VOID_CHAR)  # hold している mino
        self.hold_used = False  # 今のターンに hold したか否か
        self.current_action = None

        self.line_total_count = 0
        self.score = 0
        self.turns = 0

        # 順列をランダムに shuffle して保持
        add_permutation = list(self.minos)
        random.shuffle(add_permutation)
        for mino in add_permutation:
            self.mino_permutation.append(mino)

        self.next_mino_num = min(NEXT_MINO_NUM, len(self.mino_permutation))

        # 初期状態でミノを生成
        self.current_mino_state = self._generate_mino_state()
        self.game_over = False

    def reset(self) -> None:
        self.board = TetrisBoard(self.board.height, self.board.width, self.minos)
        self.mino_permutation = deque()
        self.hold_mino = Mino(0, np.array([[0]]), VOID_CHAR)
        self.hold_used = False
        self.line_total_count = 0
        self.score = 0
        self.turns = 0
        self.game_over = False

        add_permutation = list(self.minos)
        random.shuffle(add_permutation)
        for mino in add_permutation:
            self.mino_permutation.append(mino)

        self.current_mino_state = self._generate_mino_state()

    def observe(self) -> TetrisState:
        return TetrisState(
            self.board,
            self.current_mino_state,
            self.hold_mino,
            list(self.mino_permutation)[:NEXT_MINO_NUM],
            self.score,
            self.line_total_count,
            self.turns,
        )

    def _generate_mino_state(self) -> MinoState:
        self.turns += 1
        selected_mino = self.mino_permutation.popleft()

        # len(permutation) < 7 で新しい permutation を puh_back
        if len(self.mino_permutation) < 7:
            add_permutation = copy.deepcopy(list(self.minos))
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
        mino_state = copy.deepcopy(self.current_mino_state)
        mino_state.move(1, 0, self.board.board)
        return mino_state.origin == self.current_mino_state.origin

    def hold(self) -> None:
        self.hold_used = True
        if self.hold_mino.id == 0:
            self.hold_mino = self.current_mino_state.mino
            self.current_mino_state = self._generate_mino_state()
        else:
            self.current_mino_state, self.hold_mino = (
                MinoState(
                    mino=self.hold_mino,
                    height=self.board.height,
                    width=self.board.width,
                    origin=(
                        0,
                        self.board.width // 2 - self.hold_mino.shape.shape[1] // 2,
                    ),
                ),
                self.current_mino_state.mino,
            )

    def place(self) -> None:
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

    def step(self, actionId: int) -> None:
        action = self.action_map[actionId]
        if action.id == 0:  # move left
            self.current_mino_state.move(0, -1, self.board.board)
        elif action.id == 1:  # move right
            self.current_mino_state.move(0, 1, self.board.board)
        elif action.id == 2:  # move down
            if self.is_mino_landed():
                self.place()
            else:
                self.current_mino_state.move(1, 0, self.board.board)
        elif action.id == 3:  # rotate left
            self.current_mino_state.rotate_left(self.board.board)
        elif action.id == 4:  # rotate right
            self.current_mino_state.rotate_right(self.board.board)
        elif action.id == 5:  # hold
            if self.hold_used is False:
                self.hold()
        elif action.id == 6:  # hard drop
            while not self.is_mino_landed():
                self.current_mino_state.move(1, 0, self.board.board)
            self.place()

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
        for i in range(self.next_mino_num):
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
            for i in range(self.hold_mino.shape.shape[0]):
                s = VOID_CHAR
                if self.hold_mino.id == 4:
                    s += VOID_CHAR
                for j in range(self.hold_mino.shape.shape[1]):
                    if self.hold_mino.shape[i][j] == 1:
                        s += self.hold_mino.char
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
