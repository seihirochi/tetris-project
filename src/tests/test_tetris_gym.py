from tetris_gym import Tetrimino, TetrisBoard


def test_board_creation():
    board = TetrisBoard(10, 20)
    assert board.width == 10
    assert board.height == 20
    assert board.board.shape == (20, 10)


def test_set_value():
    board = TetrisBoard(10, 20)
    board.set_value(5, 5, 1)
    assert board.board[5][5] == 1


def test_set_tetrimino():
    board = TetrisBoard(10, 20)
    board.set_tetrimino(Tetrimino.MINO_I, 3, 3)
    assert board.board[3][3] == 1
    assert board.board[3][4] == 1
    assert board.board[3][5] == 1
    assert board.board[3][6] == 1
    board.set_tetrimino(Tetrimino.MINO_J, 5, 5)
    assert board.board[5][5] == 2
    assert board.board[6][5] == 2
    assert board.board[6][6] == 2
    assert board.board[6][7] == 2


def test_render_to_string_empty_board():
    board = TetrisBoard(10, 20)
    expected = """############
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
############
"""
    assert board.render_to_string() == expected


def test_render_to_string_with_tetrimino():
    board = TetrisBoard(10, 20)
    board.set_tetrimino(Tetrimino.MINO_I, 3, 3)
    expected = """############
#          #
#          #
#          #
#   IIII   #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#          #
############
"""
    assert board.render_to_string() == expected
