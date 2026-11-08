"""study/stats.py: metric definitions, pairing and the two resampling modes."""
import numpy as np
import pytest

from study.stats import Resampler, metrics_from_counts, percentile_interval, wilson_interval


def synthetic(n_per_class=600, block=30, clustered=False, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_per_class)
    ids = np.tile(np.arange(n_per_class), 2)
    if clustered:  # whole id blocks right or wrong together: strong within-block dependence
        block_ok = rng.random(2 * (n_per_class // block + 1)) < 0.85
        correct = block_ok[y * (n_per_class // block + 1) + ids // block]
    else:
        correct = rng.random(y.size) < 0.85
    return y, ids, np.where(correct, y, 1 - y)


def test_metric_definitions_match_hand_counts():
    m = metrics_from_counts(tn=40, fp=10, fn=5, tp=45)
    assert m["accuracy"] == pytest.approx(0.85)
    assert m["error_negative"] == pytest.approx(10 / 50) and m["error_positive"] == pytest.approx(5 / 50)
    assert m["precision_positive"] == pytest.approx(45 / 55) and m["recall_negative"] == pytest.approx(40 / 50)
    f1_pos, f1_neg = 2 * 45 / (2 * 45 + 10 + 5), 2 * 40 / (2 * 40 + 5 + 10)
    assert m["macro_f1"] == pytest.approx((f1_pos + f1_neg) / 2)


def test_a_constant_predictor_has_zero_f1_for_the_missing_class():
    m = metrics_from_counts(tn=50, fp=0, fn=50, tp=0)
    assert m["accuracy"] == 0.5 and m["macro_f1"] == pytest.approx((2 * 50 / 150) / 2)
    assert np.isnan(m["precision_positive"])


@pytest.mark.parametrize("block", [1, 30])
def test_point_estimates_do_not_depend_on_resampling(block):
    y, ids, pred = synthetic()
    r = Resampler(y, ids, block=block, n_boot=200, seed=1)
    point, reps = r.metrics(y, pred)["accuracy"]
    assert point == pytest.approx((pred == y).mean())
    assert reps.shape == (200,) and abs(reps.mean() - point) < 0.01


@pytest.mark.parametrize("block", [1, 30])
def test_identical_candidates_have_a_zero_paired_difference_in_every_replicate(block):
    y, ids, pred = synthetic()
    point, reps = Resampler(y, ids, block=block, n_boot=300, seed=2).paired(y, pred, pred)["accuracy"]
    assert point == 0 and np.all(reps == 0)


def test_paired_difference_equals_the_difference_of_marginal_point_estimates():
    y, ids, pred_a = synthetic(seed=3)
    _, _, pred_b = synthetic(seed=4)
    r = Resampler(y, ids, block=30, n_boot=100, seed=5)
    diff = r.paired(y, pred_a, pred_b)
    for metric in ("accuracy", "macro_f1", "error_negative"):
        assert diff[metric][0] == pytest.approx(r.metrics(y, pred_a)[metric][0] - r.metrics(y, pred_b)[metric][0])


def test_block_bootstrap_widens_intervals_when_errors_cluster_within_blocks():
    y, ids, pred = synthetic(n_per_class=3000, clustered=True, seed=6)
    widths = {}
    for block in (1, 30):
        lo, hi = percentile_interval(Resampler(y, ids, block=block, n_boot=2000, seed=7).metrics(y, pred)["accuracy"][1])
        widths[block] = hi - lo
    assert widths[30] > 2 * widths[1]


def test_example_level_interval_agrees_with_wilson_for_independent_errors():
    y, ids, pred = synthetic(n_per_class=5000, seed=8)
    point, reps = Resampler(y, ids, block=1, n_boot=4000, seed=9).metrics(y, pred)["accuracy"]
    lo, hi = percentile_interval(reps)
    w_lo, w_hi = wilson_interval(int(round(point * y.size)), y.size)
    assert abs(lo - w_lo) < 0.003 and abs(hi - w_hi) < 0.003


def test_masks_restrict_the_population_in_both_modes():
    y, ids, pred = synthetic(seed=10)
    mask = ids < 100
    for block in (1, 30):
        point, _ = Resampler(y, ids, block=block, n_boot=50, seed=11).metrics(y, pred, mask)["accuracy"]
        assert point == pytest.approx((pred[mask] == y[mask]).mean())
