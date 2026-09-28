import numpy as np
import pytest

from laya_demo.calibration import expected_calibration_error, plot_reliability, reliability_bins


def test_perfectly_calibrated_is_zero():
    # 10 answers at 0.8 with 8 right, 10 at 0.3 with 3 right
    conf = [0.8] * 10 + [0.3] * 10
    correct = [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_always_certain_coin_flip():
    assert expected_calibration_error([1.0] * 4, [1, 0, 1, 0]) == pytest.approx(0.5)


def test_hand_computed_weighted_gap():
    # bin (0.9, 1.0]: conf 0.95, acc 0.5, 2 items; bin (0.5, 0.6]: conf 0.55, acc 1.0, 2 items
    conf = [0.95, 0.95, 0.55, 0.55]
    correct = [1, 0, 1, 1]
    assert expected_calibration_error(conf, correct) == pytest.approx(0.5 * 0.45 + 0.5 * 0.45)


def test_bins_skip_empty_and_edges_are_right_closed():
    bins = reliability_bins([0.0, 0.1, 0.1000001, 1.0], [1, 0, 1, 1], n_bins=10)
    assert [(round(b.lo, 1), b.count) for b in bins] == [(0.0, 2), (0.1, 1), (0.9, 1)]


def test_matches_laya_own_ece():
    from laya.common import ece_score

    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 500)
    correct = (rng.uniform(0, 1, 500) < conf**2).astype(float)
    assert expected_calibration_error(conf, correct, n_bins=15) == pytest.approx(ece_score(conf, correct, bins=15))


def test_empty_is_nan_and_mismatch_raises():
    assert np.isnan(expected_calibration_error([], []))
    with pytest.raises(ValueError):
        reliability_bins([0.5], [1, 0])


def test_plot_draws_without_a_display():
    import matplotlib

    matplotlib.use("Agg")
    ax = plot_reliability([0.9, 0.8, 0.2], [1, 0, 0])
    assert "ECE" in ax.get_title()
