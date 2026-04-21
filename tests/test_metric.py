import numpy as np
import pytest
from SRA_Helper.Metric import align_and_rmse


X = np.logspace(0, 2, 50)  # positive x required for logspace common axis


def test_identical_curves_rmse_zero():
    y = np.ones(50)
    _, _, _, rmse = align_and_rmse(X, y, X, y)
    assert rmse == pytest.approx(0.0)


def test_constant_offset_rmse():
    y1 = np.ones(50)
    y2 = np.ones(50) * 3.0
    _, _, _, rmse = align_and_rmse(X, y1, X, y2)
    assert rmse == pytest.approx(2.0)


def test_unsorted_x_handled():
    rng = np.random.default_rng(0)
    idx = rng.permutation(50)
    y = np.ones(50)
    _, _, _, rmse = align_and_rmse(X[idx], y[idx], X, y)
    assert rmse == pytest.approx(0.0)


def test_no_overlap_raises():
    x1 = np.logspace(0, 1, 20)
    x2 = np.logspace(2, 3, 20)
    with pytest.raises(ValueError, match="do not overlap"):
        align_and_rmse(x1, np.ones(20), x2, np.ones(20))


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        align_and_rmse(X, np.ones(10), X, np.ones(50))


def test_2d_input_raises():
    y_2d = np.ones((5, 5))
    with pytest.raises(ValueError, match="1D"):
        align_and_rmse(y_2d, y_2d, X, np.ones(50))


def test_returns_common_x_within_overlap():
    x1 = np.logspace(0, 2, 50)
    x2 = np.logspace(1, 3, 50)
    common_x, _, _, _ = align_and_rmse(x1, np.ones(50), x2, np.ones(50))
    assert common_x.min() >= x1.min()
    assert common_x.max() <= x2.max()
