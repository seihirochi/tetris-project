import numpy as np

from tetris_gym import Mino, MinoState, Tetris, TetrisBoard


def test_board_creation():
    board = TetrisBoard(20, 10, {})
    assert board.width == 10
    assert board.height == 20
    assert board.board.shape == (20, 10)


def test_set_mino_id():
    mino_O = Mino(1, np.array([[1, 1], [1, 1]]), "O")
    board = TetrisBoard(20, 10, {mino_O})
    board.set_mino_id((5, 5), mino_O.id)
    assert board.board[5][5] == mino_O.id


def test_set_mino():
    mino_I = Mino(
        1,
        np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),
        "I",
    )
    mino_I_state = MinoState(mino_I, 20, 10, (3, 3))
    mino_O = Mino(2, np.array([[1, 1], [1, 1]]), "O")
    mino_O_state = MinoState(mino_O, 20, 10, (5, 5))

    board = TetrisBoard(20, 10, {mino_I, mino_O})
    board.set_mino(mino_I_state)

    assert board.board[3][4] == 1
    assert board.board[4][4] == 1
    assert board.board[5][4] == 1
    assert board.board[6][4] == 1
    board.set_mino(mino_O_state)
    assert board.board[5][5] == 2
    assert board.board[5][6] == 2
    assert board.board[6][5] == 2
    assert board.board[6][6] == 2
