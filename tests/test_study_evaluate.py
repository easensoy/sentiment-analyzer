"""study/evaluate.py on synthetic predictions: the analysis outputs, their internal consistency, and the scoring guards."""
import json

import numpy as np
import pandas as pd
import pytest

import study.evaluate as ev

N_PER_CLASS = 300


@pytest.fixture
def fake_study(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], N_PER_CLASS)
    ids = np.tile(np.arange(N_PER_CLASS), 2)
    pred = pd.DataFrame({"example_id": [f"test/{'pos' if l else 'neg'}/{i}" for l, i in zip(y, ids)], "label": y,
                         "review_id": ids, "rating": np.where(y == 1, rng.integers(7, 11, y.size), rng.integers(1, 5, y.size)),
                         "n_words": rng.integers(20, 600, y.size)})
    good = np.where(rng.random(y.size) < 0.9, y, 1 - y)
    worse = np.where(rng.random(y.size) < 0.8, y, 1 - y)
    for cid, p in (("majority", np.zeros_like(y)), ("good", good), ("worse", worse)):
        pred[f"score_{cid}"] = np.where(p == 1, 1.0, -1.0) if cid != "majority" else np.nan
        pred[f"pred_{cid}"] = p
    out = tmp_path / "official_test"
    out.mkdir()
    pred.to_csv(out / "predictions.csv.gz", index=False)
    overlap = pd.DataFrame({"example_id": pred["example_id"], "exact_in_train": False,
                            "normalised_in_train": np.arange(y.size) < 5, "max_cosine_to_train": 0.1})
    overlap.to_csv(tmp_path / "overlap.csv.gz", index=False)
    pred[["example_id", "label"]].iloc[::3].to_csv(tmp_path / "llm_ids.csv", index=False)
    proto = {"candidates": {"majority": {}, "good": {}, "worse": {}},
             "comparisons": [{"id": "good_vs_worse", "a": "good", "b": "worse", "factor": "test"}],
             "statistics": {"block_ids": 30, "sensitivity_block_ids": [1, 60], "n_boot": 400, "seed": 1},
             "slices": {"length_cut_points_words": [150, 300, 450]},
             "llm_baseline": {"eligible_test_ids": "llm_ids.csv"}}
    monkeypatch.setattr(ev, "protocol", lambda: proto)
    monkeypatch.setattr(ev, "PREDICTIONS", out / "predictions.csv.gz")
    monkeypatch.setattr(ev, "OUT", out)
    monkeypatch.setattr(ev, "OVERLAP", tmp_path / "overlap.csv.gz")
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    ev.analyse()
    return out, pred


def test_metrics_are_point_estimates_of_the_saved_predictions(fake_study):
    out, pred = fake_study
    m = pd.read_csv(out / "metrics.csv")
    acc = m[(m["candidate"] == "good") & (m["metric"] == "accuracy") & (m["block_ids"] == 30)].iloc[0]
    assert acc["point"] == pytest.approx((pred["pred_good"] == pred["label"]).mean())
    assert acc["lo95"] <= acc["point"] <= acc["hi95"]
    maj = m[(m["candidate"] == "majority") & (m["block_ids"] == 30)].set_index("metric")
    assert maj.loc["accuracy", "point"] == 0.5 and maj.loc["error_positive", "point"] == 1.0
    assert np.isnan(maj.loc["precision_positive", "point"])


def test_paired_rows_are_consistent_with_the_marginal_metrics(fake_study):
    out, pred = fake_study
    d = pd.read_csv(out / "paired_differences.csv")
    row = d[(d["metric"] == "accuracy") & (d["block_ids"] == 30)].iloc[0]
    expected = (pred["pred_good"] == pred["label"]).mean() - (pred["pred_worse"] == pred["label"]).mean()
    assert row["point"] == pytest.approx(expected)
    assert row["lo_bonferroni"] <= row["lo95"] <= row["point"] <= row["hi95"] <= row["hi_bonferroni"]
    assert set(d["block_ids"]) == {1, 30, 60}


def test_slices_cover_every_review_once_per_class(fake_study):
    out, pred = fake_study
    s = pd.read_csv(out / "slices.csv")
    lengths = s[(s["candidate"] == "good") & s["slice"].str.startswith("length")]
    assert lengths.groupby("true_class")["reviews"].sum().tolist() == [N_PER_CLASS, N_PER_CLASS]
    ratings = s[(s["candidate"] == "good") & s["slice"].str.startswith("rating")]
    assert ratings["reviews"].sum() == 2 * N_PER_CLASS


def test_overlap_sensitivity_and_llm_subset_use_the_frozen_populations(fake_study):
    out, _ = fake_study
    sens = pd.read_csv(out / "overlap_sensitivity.csv")
    assert set(sens["n"]) == {2 * N_PER_CLASS - 5}
    llm = pd.read_csv(out / "llm_subset_metrics.csv")
    assert set(llm["n"]) == {N_PER_CLASS * 2 // 3}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["overlap_excluded"] == 5 and summary["n_test"] == 2 * N_PER_CLASS


def test_scoring_refuses_when_the_lock_does_not_verify(monkeypatch):
    monkeypatch.setattr(ev, "verify", lambda: ["study/evaluate.py"])
    with pytest.raises(SystemExit, match="refusing to score"):
        ev.score()


def test_scoring_refuses_a_second_run(fake_study, monkeypatch):
    monkeypatch.setattr(ev, "verify", lambda: [])
    with pytest.raises(SystemExit, match="scored once already"):
        ev.score()
