"""Score the frozen candidates on the official test partition once, then analyse the saved scores.

    python -m study.evaluate score     # refuses unless the protocol lock verifies; refuses to score twice
    python -m study.evaluate analyse   # metrics, paired differences, slices and sensitivity analyses

Outputs in results/official_test/ (no review text anywhere):
  predictions.csv.gz         per test review: example_id, label, rating, n_words, review_id, and per candidate
                             the signed decision score (score_<id>; empty for the majority rule) and label (pred_<id>)
  scoring_log.json           protocol and lock hashes, git commit, timestamps, environment and hardware
  inference.csv              per candidate: batch scoring wall time over all test reviews, reviews with no
                             in-vocabulary feature, and single-request latency quantiles on a fixed sample
  metrics.csv                per candidate, metric and block length: point estimate and percentile interval
  paired_differences.csv     per comparison, metric and block length: A - B with 95% and Bonferroni intervals
  slices.csv                 per candidate, slice and true class: error rate with a 95% block-bootstrap interval
  overlap_sensitivity.csv    metrics and paired differences without test reviews that overlap training texts
  llm_subset_metrics.csv     classical candidates on the frozen 2,000-review LLM subset
  summary.json               the headline numbers the README quotes
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import environment, labels_from_scores, load_artefact
from study.protocol import LOCK, PROTOCOL, file_sha256, verify
from study.stats import Resampler, percentile_interval

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived" / "imdb_official.csv.gz"
MEMBERSHIP = ROOT / "results" / "data" / "split_membership.csv.gz"
OVERLAP = ROOT / "results" / "data" / "test_overlap.csv.gz"
OUT = ROOT / "results" / "official_test"
PREDICTIONS = OUT / "predictions.csv.gz"
GZIP = {"method": "gzip", "mtime": 0}
LATENCY_SAMPLE, LATENCY_SEED = 1_000, 7
REPORTED = ["accuracy", "macro_f1", "error_negative", "error_positive", "precision_negative", "recall_negative",
            "precision_positive", "recall_positive"]


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def hardware() -> dict:
    info = {"machine": platform.machine(), "system": f"{platform.system()} {platform.release()} ({platform.version()})",
            "logical_cpus": os.cpu_count()}
    if platform.system() == "Windows":
        try:
            query = ("$c=Get-CimInstance Win32_Processor|Select-Object -First 1;$o=Get-CimInstance Win32_OperatingSystem;"
                     "\"$($c.Name)|$($c.NumberOfCores)|$([math]::Round($o.TotalVisibleMemorySize/1MB,1))\"")
            name, cores, ram = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True,
                                              text=True, timeout=60).stdout.strip().split("|")
            info.update(cpu=name.strip(), physical_cores=int(cores), ram_gb=float(ram))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return info


def git_commit() -> str:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD") + (" (working tree differs)" if run("status", "--porcelain", "--untracked-files=no") else "")


def test_frame() -> pd.DataFrame:
    membership = pd.read_csv(MEMBERSHIP, usecols=["example_id", "partition", "label", "review_id", "rating", "n_words"])
    texts = pd.read_csv(DERIVED, usecols=["example_id", "text"])
    frame = membership[membership["partition"] == "test"].merge(texts, on="example_id", validate="one_to_one")
    return frame.sort_values("example_id").reset_index(drop=True).drop(columns="partition")


def score() -> None:
    problems = verify()
    if problems:
        raise SystemExit(f"protocol lock does not verify; refusing to score the test partition: {problems}")
    if PREDICTIONS.exists():
        raise SystemExit(f"{PREDICTIONS.relative_to(ROOT)} exists: the test partition has been scored once already")
    proto = protocol()
    frame = test_frame()
    texts, started = frame["text"].tolist(), datetime.now(timezone.utc)
    rng = np.random.default_rng(LATENCY_SEED)
    sample = [texts[i] for i in rng.choice(len(texts), LATENCY_SAMPLE, replace=False)]
    out, inference = frame.drop(columns="text").copy(), []
    for cid, spec in proto["candidates"].items():
        if spec["config"] is None:
            fit = json.loads((ROOT / "results" / "study" / "fits" / f"{cid}.json").read_text(encoding="utf-8"))
            out[f"score_{cid}"] = np.nan
            out[f"pred_{cid}"] = fit["majority_class"]
            inference.append({"candidate": cid, "batch_seconds": 0.0, "zero_feature_reviews": 0})
            continue
        artefact_dir = ROOT / ("models/artefacts" if cid == "svc_uni_8k_historical" else "models/study/artefacts") / spec["model_version"]
        pipeline, manifest = load_artefact(artefact_dir)
        if manifest["artefact_sha256"] != spec["artefact_sha256"]:
            raise SystemExit(f"{cid}: artefact differs from the frozen protocol")
        features, classifier = pipeline[:-1], pipeline[-1]
        start = time.perf_counter()
        x = features.transform(texts)
        scores = classifier.decision_function(x)
        batch = time.perf_counter() - start
        out[f"score_{cid}"], out[f"pred_{cid}"] = scores, labels_from_scores(scores)
        latencies = []
        for text in sample:  # one request at a time: cleaning + TF-IDF + one decision_function call
            t0 = time.perf_counter()
            classifier.decision_function(features.transform([text]))
            latencies.append((time.perf_counter() - t0) * 1000)
        q = np.quantile(latencies, [0.5, 0.9, 0.99])
        inference.append({"candidate": cid, "batch_seconds": round(batch, 3),
                          "batch_reviews_per_second": round(len(texts) / batch, 1),
                          "zero_feature_reviews": int((np.diff(x.indptr) == 0).sum()),
                          "latency_ms_p50": round(q[0], 3), "latency_ms_p90": round(q[1], 3), "latency_ms_p99": round(q[2], 3),
                          "latency_sample": LATENCY_SAMPLE})
        print(f"{cid}: scored {len(texts):,} reviews in {batch:.1f} s; p50 latency {q[0]:.2f} ms")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS, index=False, lineterminator="\n", compression=GZIP, float_format="%.17g")
    pd.DataFrame(inference).to_csv(OUT / "inference.csv", index=False, lineterminator="\n")
    log = {"protocol_sha256": file_sha256(PROTOCOL), "lock_sha256": file_sha256(LOCK), "git_commit": git_commit(),
           "scoring_started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "scoring_finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_test_reviews": len(texts), "environment": environment(), "hardware": hardware(),
           "timing_note": "wall-clock in one process while no other study job ran; other software on the shared "
                          "machine was not controlled; single-request latency excludes HTTP and JSON handling"}
    (OUT / "scoring_log.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {PREDICTIONS.relative_to(ROOT)}, inference.csv, scoring_log.json")


def interval_rows(stats: dict, keys: dict, level_names=("lo95", "hi95"), bonferroni: float | None = None) -> list[dict]:
    rows = []
    for metric in REPORTED:
        point, reps = stats[metric]
        lo, hi = percentile_interval(reps)
        row = {**keys, "metric": metric, "point": point, level_names[0]: lo, level_names[1]: hi}
        if bonferroni:
            row["lo_bonferroni"], row["hi_bonferroni"] = percentile_interval(reps, 1 - 0.05 / bonferroni)
            finite = reps[np.isfinite(reps)]
            row["share_of_replicates_above_0"] = float((finite > 0).mean())
        rows.append(row)
    return rows


def analyse() -> None:
    proto = protocol()
    stat = proto["statistics"]
    pred = pd.read_csv(PREDICTIONS)
    y, ids = pred["label"].to_numpy(), pred["review_id"].to_numpy()
    cids = list(proto["candidates"])
    comps = proto["comparisons"]
    resamplers = {b: Resampler(y, ids, block=b, n_boot=stat["n_boot"], seed=stat["seed"])
                  for b in [stat["block_ids"], *stat["sensitivity_block_ids"]]}
    metric_rows, paired_rows = [], []
    for block, r in resamplers.items():
        for cid in cids:
            metric_rows += interval_rows(r.metrics(y, pred[f"pred_{cid}"].to_numpy()), {"candidate": cid, "block_ids": block})
        for comp in comps:
            diff = r.paired(y, pred[f"pred_{comp['a']}"].to_numpy(), pred[f"pred_{comp['b']}"].to_numpy())
            paired_rows += interval_rows(diff, {"comparison": comp["id"], "a": comp["a"], "b": comp["b"],
                                                "factor": comp["factor"], "block_ids": block}, bonferroni=len(comps))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(OUT / "metrics.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.DataFrame(paired_rows).to_csv(OUT / "paired_differences.csv", index=False, lineterminator="\n", float_format="%.6g")

    primary = resamplers[stat["block_ids"]]
    c1, c2, c3 = proto["slices"]["length_cut_points_words"]
    words = pred["n_words"]
    slice_defs = [(f"length <= {c1} words", words <= c1), (f"length {c1 + 1}-{c2} words", (words > c1) & (words <= c2)),
                  (f"length {c2 + 1}-{c3} words", (words > c2) & (words <= c3)), (f"length > {c3} words", words > c3)]
    slice_defs += [(f"rating {r}", pred["rating"] == r) for r in sorted(pred["rating"].unique())]
    slice_rows = []
    for name, mask in slice_defs:
        for cid in cids:
            m = primary.metrics(y, pred[f"pred_{cid}"].to_numpy(), mask.to_numpy())
            for cls in ("negative", "positive"):
                point, reps = m[f"error_{cls}"]
                if not np.isfinite(point):
                    continue
                lo, hi = percentile_interval(reps)
                n = int((mask & (pred["label"] == (cls == "positive"))).sum())
                slice_rows.append({"slice": name, "candidate": cid, "true_class": cls, "reviews": n,
                                   "errors": int(round(point * n)), "error_rate": point, "lo95": lo, "hi95": hi})
    pd.DataFrame(slice_rows).to_csv(OUT / "slices.csv", index=False, lineterminator="\n", float_format="%.6g")

    overlap = pd.read_csv(OVERLAP)
    flags = pred[["example_id"]].merge(overlap, on="example_id", how="left", validate="one_to_one")
    keep = ~(flags["normalised_in_train"].astype(bool) | (flags["max_cosine_to_train"] >= 0.9)).to_numpy()
    sens = []
    for cid in cids:
        sens += interval_rows(primary.metrics(y, pred[f"pred_{cid}"].to_numpy(), keep),
                              {"population": "test minus overlap", "n": int(keep.sum()), "candidate": cid})
    for comp in comps:
        sens += interval_rows(primary.paired(y, pred[f"pred_{comp['a']}"].to_numpy(), pred[f"pred_{comp['b']}"].to_numpy(), keep),
                              {"population": "test minus overlap", "n": int(keep.sum()), "comparison": comp["id"]},
                              bonferroni=len(comps))
    pd.DataFrame(sens).to_csv(OUT / "overlap_sensitivity.csv", index=False, lineterminator="\n", float_format="%.6g")

    llm_ids = set(pd.read_csv(ROOT / proto["llm_baseline"]["eligible_test_ids"])["example_id"])
    in_llm = pred["example_id"].isin(llm_ids).to_numpy()
    llm_rows = []
    for cid in cids:
        llm_rows += interval_rows(primary.metrics(y, pred[f"pred_{cid}"].to_numpy(), in_llm),
                                  {"population": "LLM subset", "n": int(in_llm.sum()), "candidate": cid})
    pd.DataFrame(llm_rows).to_csv(OUT / "llm_subset_metrics.csv", index=False, lineterminator="\n", float_format="%.6g")

    metrics = pd.DataFrame(metric_rows)
    head = metrics[metrics["block_ids"] == stat["block_ids"]].set_index(["candidate", "metric"])
    summary = {"n_test": int(len(pred)), "block_ids": stat["block_ids"], "n_boot": stat["n_boot"],
               "overlap_excluded": int((~keep).sum()),
               "candidates": {cid: {m: [round(head.loc[(cid, m), k], 6) for k in ("point", "lo95", "hi95")]
                                    for m in ("accuracy", "macro_f1", "error_negative", "error_positive")} for cid in cids}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["score", "analyse"])
    {"score": score, "analyse": analyse}[parser.parse_args().command]()


if __name__ == "__main__":
    sys.exit(main())
