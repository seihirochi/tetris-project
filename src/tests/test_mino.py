import numpy as np
import pytest

from tetris_gym import Mino

mock_mino_shape = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]])
mock_mino_shape_2 = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]])


class TestMino:
    def test_success_creation(self):
        valid_mino_j = Mino(1, mock_mino_shape)
        assert valid_mino_j.shape.shape == (3, 3)
        assert valid_mino_j.char == "▪︎"
        assert np.array_equal(valid_mino_j.shape, mock_mino_shape)

        valid_mino_l = Mino(1, mock_mino_shape_2)
        assert valid_mino_l.shape.shape == (3, 3)
        assert valid_mino_l.char == "▪︎"
        assert np.array_equal(valid_mino_l.shape, mock_mino_shape_2)

    def test_failcreation__with_invalid_shape(self):
        with pytest.raises(ValueError):
            Mino(1, np.array([[0, 0, 1], [0, 0, 1]]))

    def test_failcreation__with_empty_char(self):
        with pytest.raises(ValueError):
            Mino(1, mock_mino_shape, "")
