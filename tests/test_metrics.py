import numpy as np
import pytest

from src.evaluation import metrics as M

UNIFORM = np.full((3, 3), 1 / 3)
Y_ALL = np.array([0, 1, 2])


def test_rps_uniform_known_values():
    got = M.rps_per_match(UNIFORM, Y_ALL)
    assert got == pytest.approx([5 / 18, 1 / 9, 5 / 18])


def test_rps_perfect_is_zero():
    assert M.rps(np.eye(3), Y_ALL) == pytest.approx(0.0)


def test_rps_penalises_distance():
    pred_draw = np.array([[0.0, 1.0, 0.0]])
    pred_away = np.array([[0.0, 0.0, 1.0]])
    assert M.rps(pred_away, np.array([0])) > M.rps(pred_draw, np.array([0]))


def test_log_loss_and_brier_uniform():
    assert M.log_loss(UNIFORM, Y_ALL) == pytest.approx(np.log(3))
    assert M.brier(UNIFORM, Y_ALL) == pytest.approx(2 / 3)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        M.rps(np.array([[0.5, 0.5, 0.5]]), np.array([0]))
