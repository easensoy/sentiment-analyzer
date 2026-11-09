"""The benchmark harness's arithmetic, its load generator, and the consistency of the committed benchmark results."""
import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from study import benchmark, report

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "results" / "benchmarks"


# ----------------------------------------------------------------------------- harness logic

def test_summarise_level_counts_failures_and_drops_requests_that_end_after_the_window():
    records = [(0.0, 0.010, True, "200"), (0.5, 0.52, False, "500"), (1.0, 1.03, True, "200"),
               (14.99, 15.2, True, "200")]
    s = benchmark.summarise_level(records, measure_s=15.0)
    assert (s["completed"], s["failed"]) == (3, 1)
    assert s["failure_rate"] == pytest.approx(1 / 3)
    assert s["throughput_rps"] == pytest.approx(2 / 15)
    assert s["latency_ms_p50"] == pytest.approx(20.0)


def test_closed_loop_records_successes_errors_and_exceptions_and_marks_the_window():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.002)  # a real await, so workers interleave as they do over a socket
        text = json.loads(request.content)["text"]
        if text == "raise":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(500 if text == "fail" else 200, json={"label": "positive"})

    marks = []
    records = asyncio.run(benchmark.closed_loop("http://bench", ["ok", "fail", "raise"], concurrency=3,
                                                warmup_s=0.05, measure_s=0.3, timeout_s=5,
                                                transport=httpx.MockTransport(handler), on_window=marks.append))
    assert marks == ["start", "end"]
    assert {status for *_, status in records} == {"200", "500", "ConnectError"}
    assert all(start >= 0 for start, *_ in records)
    assert all(ok == (status == "200") for _, _, ok, status in records)
    s = benchmark.summarise_level(records, 0.3)
    assert 0.5 < s["failure_rate"] < 0.8  # two of the three workers always fail


def test_memory_counters_are_positive_and_peak_is_at_least_current():
    current, peak = benchmark.memory_bytes()
    assert 0 < current <= peak


def test_costs_follow_the_formula_and_use_the_peak_throughput_level(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "OUT", tmp_path)
    pd.DataFrame([{"candidate": "m", "cpu_s_per_1000_batch": 2.0}]).to_csv(tmp_path / "inference.csv", index=False)
    pd.DataFrame([{"candidate": "m", "sweep": s, "concurrency": c, "throughput_rps": t, "server_cpu_s_per_1000": cpu}
                  for s, c, t, cpu in [(0, 1, 100, 9.0), (0, 2, 150, 4.0), (1, 1, 100, 9.0), (1, 2, 140, 6.0)]]
                 ).to_csv(tmp_path / "http.csv", index=False)
    benchmark.run_costs()
    c = pd.read_csv(tmp_path / "costs.csv")
    assert len(c) == 2 * len(benchmark.RATES_USD_PER_VCPU_HOUR) * len(benchmark.UTILISATION)
    assert np.allclose(c["usd_per_1000_predictions"],
                       c["rate_usd_per_vcpu_hour"] * c["cpu_s_per_1000"] / 3600 / c["utilisation"])
    interactive = c[c["mode"].str.startswith("interactive")]["cpu_s_per_1000"].unique()
    assert interactive.tolist() == [5.0]  # median over sweeps of the CPU at each sweep's peak level


def test_pareto_efficiency_requires_strict_improvement_in_one_dimension():
    assert report.pareto_efficient([1, 2, 3, 4, 5], [0.80, 0.85, 0.84, 0.90, 0.90]) == [True, True, False, True, False]
    assert report.pareto_efficient([1, 1], [0.8, 0.8]) == [True, True]


def test_significant_figure_formatting():
    assert [report.sig(v) for v in (0.1134, 7.534, 18.24, 1234.5, 0.003104)] == ["0.113", "7.53", "18.2", "1,234",
                                                                                  "0.00310"]


# ----------------------------------------------------------------------------- committed benchmark results

@pytest.fixture(scope="module")
def results():
    return {name: pd.read_csv(BENCH / f"{name}.csv") for name in ("training_runs", "inference", "http", "costs")}


def test_the_workload_is_drawn_from_the_training_partition_only():
    ids = pd.read_csv(BENCH / "workload_ids.csv")["example_id"]
    membership = pd.read_csv(ROOT / "results" / "data" / "split_membership.csv.gz", usecols=["example_id", "partition"])
    partition = membership.set_index("example_id").loc[ids, "partition"]
    assert len(ids) == benchmark.WORKLOAD_SIZE and ids.is_unique and (partition == "train").all()


def test_every_trained_configuration_was_benchmarked_with_the_planned_repeats(results):
    runs = results["training_runs"].groupby("candidate").size()
    for cid in report.trained():
        expected = benchmark.SLOW_FIT_REPEATS if cid == "svc_uni_full" else benchmark.TRAIN_REPEATS
        assert runs[cid] == expected, cid
    assert set(results["inference"]["candidate"]) == set(report.trained())


def test_training_refits_reproduce_the_frozen_artefacts(results):
    runs = results["training_runs"].set_index("candidate")
    inf = results["inference"].set_index("candidate")
    for cid in report.trained():
        assert (runs.loc[[cid], "n_features"] == inf.loc[cid, "n_features"]).all(), cid
        # the historical artefact was fitted in train.py's row order, and libsvm's solution depends slightly on
        # order; elsewhere only floating-point summation order differs between the two fits
        tolerance = 1e-3 if cid == "svc_uni_8k_historical" else 1e-6
        assert runs.loc[[cid], "max_abs_score_difference_vs_frozen"].max() <= tolerance, cid


def test_timings_are_ordered_and_positive(results):
    inf = results["inference"]
    assert (inf["latency_ms_p50"] <= inf["latency_ms_p95"]).all() and (inf["latency_ms_p95"] <= inf["latency_ms_p99"]).all()
    assert (inf["cpu_s_per_1000_batch"] == inf["cpu_s_per_1000_b1024"]).all() and (inf["cpu_s_per_1000_batch"] > 0).all()
    assert ((inf["cpu_s_per_1000_b1024_min"] <= inf["cpu_s_per_1000_b1024"])
            & (inf["cpu_s_per_1000_b1024"] <= inf["cpu_s_per_1000_b1024_max"])).all()
    http = results["http"]
    assert (http["latency_ms_p50"] <= http["latency_ms_p95"]).all()
    assert (http["completed"] > 0).all() and (http["throughput_rps"] > 0).all()
    runs = results["training_runs"]
    parts = runs[["cleaning_s", "vectoriser_fit_transform_s", "estimator_fit_s"]].sum(axis=1)
    assert np.allclose(parts, runs["pipeline_fit_s"], rtol=1e-3)


def test_the_http_sweep_is_complete(results):
    http = results["http"]
    for cid in benchmark.HTTP_CANDIDATES:
        part = http[http["candidate"] == cid]
        assert sorted(part["concurrency"].unique()) == list(benchmark.CONCURRENCY), cid
        assert len(part) == len(benchmark.CONCURRENCY) * benchmark.HTTP_SWEEPS, cid


def test_cost_estimates_use_the_measured_cpu_times(results):
    c = results["costs"]
    inf = results["inference"].set_index("candidate")
    batch = c[c["mode"].str.startswith("batch")].set_index("candidate")["cpu_s_per_1000"]
    assert np.allclose(batch.groupby(level=0).first().loc[inf.index], inf["cpu_s_per_1000_batch"])
    assert np.allclose(c["usd_per_1000_predictions"],
                       c["rate_usd_per_vcpu_hour"] * c["cpu_s_per_1000"] / 3600 / c["utilisation"])
    assert c["status"].str.startswith("estimate").all()


def test_benchmarks_ran_single_threaded():
    env = json.loads((BENCH / "environment.json").read_text(encoding="utf-8"))
    assert set(env["thread_settings"].values()) == {"1"}
    assert {"train", "inference", "http"} <= env.keys()


def test_inference_summary_matches_the_recorded_passes(results):
    passes = pd.read_csv(BENCH / "inference_passes.csv")
    assert (passes.groupby(["candidate", "batch_size"]).size() == benchmark.BATCH_REPEATS).all()
    inf = results["inference"].set_index("candidate")
    per_1000 = (1000 * passes["cpu_s"] / passes["reviews"]).groupby([passes["candidate"], passes["batch_size"]])
    for (cid, size), values in per_1000:
        assert inf.loc[cid, f"cpu_s_per_1000_b{size}"] == pytest.approx(values.median(), rel=1e-4), (cid, size)
        assert inf.loc[cid, f"cpu_s_per_1000_b{size}_min"] == pytest.approx(values.min(), rel=1e-4), (cid, size)
        assert inf.loc[cid, f"cpu_s_per_1000_b{size}_max"] == pytest.approx(values.max(), rel=1e-4), (cid, size)
