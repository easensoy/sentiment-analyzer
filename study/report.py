"""Generate the result tables in README.md and docs/ from the committed result files, or check that they are current.

    python -m study.report           # rewrite every <!-- table:NAME --> ... <!-- /table:NAME --> block
    python -m study.report --check   # exit 1 if any block differs from what the result files produce (CI runs this)

Tables are produced only from files under results/ and protocol/, so the documents cannot drift from the
measurements. Numbers quoted outside these blocks are checked by tests/test_report.py.
"""
import argparse
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCUMENTS = (README, ROOT / "docs" / "RESULTS.md", ROOT / "docs" / "BENCHMARKS.md")
RESULTS = ROOT / "results"
BENCH = RESULTS / "benchmarks"
PROTOCOL = ROOT / "protocol" / "protocol_v1.json"
ESTIMATOR = {"svc": "SVC (linear kernel)", "linear_svc": "LinearSVC", "logreg": "LogisticRegression"}
SHORT_ESTIMATOR = {"svc": "SVC", "linear_svc": "LinearSVC", "logreg": "LogReg"}
FEATURES = {(1, 1): "unigrams", (1, 2): "1–2-grams"}
# Reference point of the cost model; the sensitivity range spans every rate and utilisation in costs.csv.
REFERENCE_RATE_USD_PER_VCPU_HOUR, REFERENCE_UTILISATION = 0.05, 0.50


# ----------------------------------------------------------------------------- formatting

def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}"


def pp(v: float) -> str:
    return f"{v * 100:+.2f}".replace("-", "−")


def interval(lo: float, hi: float, fmt=pct) -> str:
    return f"{fmt(lo)}–{fmt(hi)}" if fmt is pct else f"[{fmt(lo)}, {fmt(hi)}]"


def sig(v: float, digits: int = 3) -> str:
    """`digits` significant figures in fixed notation with thousands separators (0.113, 7.53, 18.2, 1,240)."""
    if v == 0 or not math.isfinite(v):
        return "0" if v == 0 else "—"
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(v))))
    return f"{v:,.{decimals}f}"


def ratio(v: float) -> str:
    return f"{sig(v, 3 if v >= 10 else 2)}×"


# ----------------------------------------------------------------------------- inputs

@lru_cache(maxsize=None)
def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def trained() -> list[str]:
    return [c for c, spec in protocol()["candidates"].items() if spec["config"] is not None]


def describe(spec: dict) -> tuple[str, str, str]:
    cfg = spec["config"]
    if cfg is None:
        return "constant", "—", "—"
    feats = FEATURES[tuple(cfg["ngram_range"])]
    return ESTIMATOR[cfg["estimator"]], f"{feats}, {cfg['cleaning']}", f"{cfg['C']:g}"


def model_label(spec: dict) -> str:
    cfg = spec["config"]
    if cfg is None:
        return "majority class"
    return (f"{SHORT_ESTIMATOR[cfg['estimator']]}, {FEATURES[tuple(cfg['ngram_range'])]}, {cfg['cleaning']}, "
            f"C={cfg['C']:g}, {spec['n_train'] // 1000}k")


@lru_cache(maxsize=None)
def official_metrics() -> pd.DataFrame:
    m = pd.read_csv(RESULTS / "official_test" / "metrics.csv")
    return m[m["block_ids"] == protocol()["statistics"]["block_ids"]].set_index(["candidate", "metric"])


@lru_cache(maxsize=None)
def paired_accuracy() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "official_test" / "paired_differences.csv")
    return d[d["metric"] == "accuracy"].set_index(["comparison", "block_ids"])


@lru_cache(maxsize=None)
def training() -> pd.DataFrame:
    """Per configuration: medians (and ranges) over the benchmark's fresh-process training runs."""
    runs = pd.read_csv(BENCH / "training_runs.csv")
    return runs.groupby("candidate").agg(
        runs=("repeat", "size"), n_train=("n_train", "first"), n_features=("n_features", "first"),
        fit_s=("pipeline_fit_s", "median"), fit_s_min=("pipeline_fit_s", "min"), fit_s_max=("pipeline_fit_s", "max"),
        fit_cpu_s=("pipeline_fit_cpu_s", "median"), cleaning_s=("cleaning_s", "median"),
        vectoriser_s=("vectoriser_fit_transform_s", "median"), estimator_s=("estimator_fit_s", "median"),
        peak_mb=("peak_memory_mb", "median"), increment_mb=("fit_memory_increment_mb", "median"),
        increment_mb_min=("fit_memory_increment_mb", "min"), increment_mb_max=("fit_memory_increment_mb", "max"),
        score_gap=("max_abs_score_difference_vs_frozen", "max"), other_cpus=("other_busy_cpus", "median"))


@lru_cache(maxsize=None)
def inference() -> pd.DataFrame:
    return pd.read_csv(BENCH / "inference.csv").set_index("candidate")


@lru_cache(maxsize=None)
def http_levels() -> pd.DataFrame:
    """Per configuration and concurrency level: medians over sweeps, failures summed."""
    h = pd.read_csv(BENCH / "http.csv")
    return h.groupby(["candidate", "concurrency"]).agg(
        sweeps=("sweep", "size"), throughput=("throughput_rps", "median"), throughput_min=("throughput_rps", "min"),
        throughput_max=("throughput_rps", "max"), p50=("latency_ms_p50", "median"), p95=("latency_ms_p95", "median"),
        p99=("latency_ms_p99", "median"), completed=("completed", "sum"), failed=("failed", "sum"),
        server_cpu_util=("server_cpu_utilisation", "median"), client_cpu_util=("client_cpu_utilisation", "median"),
        server_cpu_s_per_1000=("server_cpu_s_per_1000", "median"), ready_s=("server_ready_s", "median")).reset_index()


def http_peak() -> pd.DataFrame:
    """The concurrency level with the highest median throughput, per served configuration."""
    levels = http_levels()
    return levels.loc[levels.groupby("candidate")["throughput"].idxmax()].set_index("candidate")


@lru_cache(maxsize=None)
def costs() -> pd.DataFrame:
    return pd.read_csv(BENCH / "costs.csv")


@lru_cache(maxsize=None)
def environment() -> dict:
    return json.loads((BENCH / "environment.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- derived quantities

def pareto_efficient(cost, value) -> list[bool]:
    """True for each point that no other point dominates (no more costly and no less valuable, one strictly)."""
    points = list(zip(cost, value))
    return [not any(c2 <= c and v2 >= v and (c2 < c or v2 > v) for c2, v2 in points) for c, v in points]


def frontier() -> pd.DataFrame:
    """Official-test accuracy against batch inference CPU time; Pareto membership is computed post hoc."""
    m, inf, proto = official_metrics(), inference(), protocol()
    rows = [{"candidate": cid, "estimator": proto["candidates"][cid]["config"]["estimator"],
             "n_train": proto["candidates"][cid]["n_train"], "accuracy": m.loc[(cid, "accuracy"), "point"],
             "lo": m.loc[(cid, "accuracy"), "lo95"], "hi": m.loc[(cid, "accuracy"), "hi95"],
             "cpu_s_per_1000": inf.loc[cid, "cpu_s_per_1000_batch"], "cpu_lo": inf.loc[cid, "cpu_s_per_1000_b1024_min"],
             "cpu_hi": inf.loc[cid, "cpu_s_per_1000_b1024_max"]} for cid in trained()]
    df = pd.DataFrame(rows).sort_values("cpu_s_per_1000").reset_index(drop=True)
    df["pareto"] = pareto_efficient(df["cpu_s_per_1000"], df["accuracy"])
    return df


def tradeoffs() -> pd.DataFrame:
    """Per pre-specified comparison: paired accuracy difference and the ratios of training and inference cost.
    Ratio ranges pair every run of A with every run of B."""
    runs = pd.read_csv(BENCH / "training_runs.csv").groupby("candidate")
    d, inf, block = paired_accuracy(), inference(), protocol()["statistics"]["block_ids"]
    rows = []
    for comp in protocol()["comparisons"]:
        a, b = runs.get_group(comp["a"]), runs.get_group(comp["b"])
        row = {"comparison": comp["id"], "a": comp["a"], "b": comp["b"], "factor": comp["factor"],
               "d_acc": d.loc[(comp["id"], block), "point"], "d_lo": d.loc[(comp["id"], block), "lo95"],
               "d_hi": d.loc[(comp["id"], block), "hi95"],
               "cpu_ratio": inf.loc[comp["a"], "cpu_s_per_1000_batch"] / inf.loc[comp["b"], "cpu_s_per_1000_batch"]}
        for key, column in (("time", "pipeline_fit_s"), ("memory", "fit_memory_increment_mb")):
            row[f"{key}_ratio"] = a[column].median() / b[column].median()
            row[f"{key}_ratio_lo"] = a[column].min() / b[column].max()
            row[f"{key}_ratio_hi"] = a[column].max() / b[column].min()
        rows.append(row)
    return pd.DataFrame(rows)


def cost_summary() -> pd.DataFrame:
    """USD per million predictions at the reference rate and utilisation, with the range over the whole grid."""
    c = costs()
    ref = c[(c["rate_usd_per_vcpu_hour"] == REFERENCE_RATE_USD_PER_VCPU_HOUR) & (c["utilisation"] == REFERENCE_UTILISATION)]
    grid = c.groupby(["candidate", "mode"])["usd_per_million_predictions"].agg(["min", "max"])
    out = ref.set_index(["candidate", "mode"])[["cpu_s_per_1000", "usd_per_million_predictions"]].join(grid)
    return out.rename(columns={"usd_per_million_predictions": "usd_reference", "min": "usd_min", "max": "usd_max"})


# ----------------------------------------------------------------------------- README tables

def table_summary() -> str:
    m, fits, inf = official_metrics(), training(), inference()
    efficient = frontier().set_index("candidate")["pareto"]
    rows = ["| Configuration | Model | Accuracy % [95% CI] | Error, negative / positive % | Fit, s | Fit memory, MB "
            "| CPU s per 1,000 predictions | Single-request p50 / p95, ms |",
            "|---|---|---|---|---:|---:|---:|---|"]
    for cid, spec in protocol()["candidates"].items():
        acc = (f"{pct(m.loc[(cid, 'accuracy'), 'point'], 2)} [{pct(m.loc[(cid, 'accuracy'), 'lo95'], 2)}–"
               f"{pct(m.loc[(cid, 'accuracy'), 'hi95'], 2)}]")
        err = f"{pct(m.loc[(cid, 'error_negative'), 'point'], 2)} / {pct(m.loc[(cid, 'error_positive'), 'point'], 2)}"
        if spec["config"] is None:
            rows.append(f"| `{cid}` | {model_label(spec)} | {acc} | {err} | — | — | — | — |")
            continue
        name = f"**`{cid}`**" if efficient[cid] else f"`{cid}`"
        rows.append(f"| {name} | {model_label(spec)} | {acc} | {err} | {sig(fits.loc[cid, 'fit_s'])} "
                    f"| {fits.loc[cid, 'increment_mb']:,.0f} | {sig(inf.loc[cid, 'cpu_s_per_1000_batch'])} "
                    f"| {inf.loc[cid, 'latency_ms_p50']:.2f} / {inf.loc[cid, 'latency_ms_p95']:.2f} |")
    return "\n".join(rows)


def table_serving() -> str:
    levels, peak, inf = http_levels(), http_peak(), inference()
    rows = ["| Configuration | Model-only p50, ms | HTTP p50 / p95 at concurrency 1, ms | Peak throughput, requests/s "
            "(concurrency) | HTTP p50 / p95 at peak, ms | Server CPU s per 1,000 requests | Failed / completed |",
            "|---|---:|---|---:|---|---:|---|"]
    for cid in protocol()["candidates"]:
        if cid not in peak.index:
            continue
        one = levels[(levels["candidate"] == cid) & (levels["concurrency"] == 1)].iloc[0]
        p = peak.loc[cid]
        part = levels[levels["candidate"] == cid]
        rows.append(f"| `{cid}` | {inf.loc[cid, 'latency_ms_p50']:.2f} | {one['p50']:.1f} / {one['p95']:.1f} "
                    f"| {p['throughput']:,.0f} ({int(p['concurrency'])}) | {p['p50']:.1f} / {p['p95']:.1f} "
                    f"| {sig(p['server_cpu_s_per_1000'])} | {int(part['failed'].sum())} / {int(part['completed'].sum()):,} |")
    return "\n".join(rows)


def table_economics() -> str:
    s = cost_summary()
    mode_name = lambda mode: "batch, in-process" if mode.startswith("batch") else "interactive HTTP"
    rows = ["| Configuration | Mode | CPU s per 1,000 predictions (measured) | USD per million predictions, reference "
            "(estimate) | Range over rates and utilisations (estimate) |", "|---|---|---:|---:|---|"]
    for cid in trained():
        for (c, mode), r in s.loc[[cid]].sort_index(level="mode").iterrows():
            rows.append(f"| `{cid}` | {mode_name(mode)} | {sig(r['cpu_s_per_1000'])} | {sig(r['usd_reference'], 2)} "
                        f"| {sig(r['usd_min'], 2)}–{sig(r['usd_max'], 2)} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- docs/RESULTS.md tables

def table_results() -> str:
    m = official_metrics()
    rows = ["| Candidate | Estimator | Features, cleaning | C | Training reviews | Accuracy % [95% CI] | Macro F1 % "
            "[95% CI] | Error, negative class % | Error, positive class % |",
            "|---|---|---|---:|---:|---|---|---|---|"]
    for cid, spec in protocol()["candidates"].items():
        est, feats, c = describe(spec)
        cell = lambda k: (f"{pct(m.loc[(cid, k), 'point'], 2)} [{pct(m.loc[(cid, k), 'lo95'], 2)}–"
                          f"{pct(m.loc[(cid, k), 'hi95'], 2)}]")
        rows.append(f"| `{cid}` | {est} | {feats} | {c} | {spec['n_train']:,} | {cell('accuracy')} | {cell('macro_f1')} "
                    f"| {cell('error_negative')} | {cell('error_positive')} |")
    return "\n".join(rows)


def table_paired() -> str:
    proto = protocol()
    d = paired_accuracy()
    main, (iid, wide) = proto["statistics"]["block_ids"], proto["statistics"]["sensitivity_block_ids"]
    rows = ["| Comparison (A − B) | Factor | Δ accuracy, pp | 95% CI, 30-id blocks | Bonferroni CI (7) "
            "| 95% CI, example-level | 95% CI, 60-id blocks |", "|---|---|---:|---|---|---|---|"]
    for comp in proto["comparisons"]:
        r = d.loc[(comp["id"], main)]
        rows.append(f"| `{comp['a']}` − `{comp['b']}` | {comp['factor']} | {pp(r['point'])} "
                    f"| {interval(r['lo95'], r['hi95'], pp)} | {interval(r['lo_bonferroni'], r['hi_bonferroni'], pp)} "
                    f"| {interval(d.loc[(comp['id'], iid), 'lo95'], d.loc[(comp['id'], iid), 'hi95'], pp)} "
                    f"| {interval(d.loc[(comp['id'], wide), 'lo95'], d.loc[(comp['id'], wide), 'hi95'], pp)} |")
    return "\n".join(rows)


def table_development() -> str:
    sel = json.loads((RESULTS / "dev" / "selection.json").read_text(encoding="utf-8"))
    rows = ["| Estimator | Features, cleaning | Selected C | Mean CV accuracy % (SD across 5 folds) |",
            "|---|---|---:|---|"]
    names = {"logreg": "LogisticRegression", "linear_svc": "LinearSVC"}
    features = {"legacy-v1/unigram": "unigrams, legacy-v1", "legacy-v1/uni+bigram": "1–2-grams, legacy-v1",
                "v2/uni+bigram": "1–2-grams, v2"}
    for est in ("linear_svc", "logreg"):
        grid = sel["grids"][est]
        for feat, label in features.items():
            s = sel["summary"][est][feat]
            edge = " (grid edge)" if s["C"] in (min(grid), max(grid)) else ""
            rows.append(f"| {names[est]} | {label} | {s['C']:g}{edge} | {pct(s['mean_accuracy'], 2)} "
                        f"({pct(s['sd_accuracy'], 2)}) |")
    ref = RESULTS / "dev" / "cv_svc_reference.csv"
    if ref.exists():
        acc = pd.read_csv(ref)["accuracy"]
        rows.append(f"| SVC (linear kernel), reference only | unigrams, legacy-v1 | 1 (not tuned) | "
                    f"{pct(acc.mean(), 2)} ({pct(acc.std(), 2)}) |")
    return "\n".join(rows)


def table_ablation() -> str:
    sel = json.loads((RESULTS / "dev" / "selection.json").read_text(encoding="utf-8"))
    abl = sel["v2_ablation_at_selected_C"]
    full = abl["v2"]["mean_accuracy"]
    rows = ["| Cleaning (1–2-grams, branch estimator at its selected C) | Mean CV accuracy % | Δ vs full v2, pp |",
            "|---|---:|---:|"]
    for name in sorted(abl, key=lambda k: (k != "v2", k)):
        rows.append(f"| {name} | {pct(abl[name]['mean_accuracy'], 2)} | "
                    f"{'—' if name == 'v2' else pp(abl[name]['mean_accuracy'] - full)} |")
    return "\n".join(rows)


def table_behavioural() -> str:
    b = pd.read_csv(RESULTS / "behavioural" / "failure_rates.csv")
    b = b[(b["split"] == "heldout") & (b["ambiguity"] == "all")].set_index(["candidate", "phenomenon"])
    phenomena = ["negation", "contrast", "sarcasm", "intensity"]
    support = {p: b.xs(p, level="phenomenon").iloc[0] for p in phenomena}
    head = " | ".join(f"{p.capitalize()} ({int(support[p]['units'])} units, {int(support[p]['families'])} families)"
                      for p in phenomena)
    rows = [f"| Candidate | {head} |", "|---|" + "---|" * len(phenomena)]
    for cid in protocol()["candidates"]:
        cells = []
        for p in phenomena:
            r = b.loc[(cid, p)]
            cells.append(f"{pct(r['failure_rate'], 0)} ({int(r['failed_units'])}/{int(r['units'])}) "
                         f"[{pct(r['lo95'], 0)}–{pct(r['hi95'], 0)}]")
        rows.append(f"| `{cid}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_overlap() -> str:
    s = pd.read_csv(RESULTS / "official_test" / "overlap_sensitivity.csv")
    m = official_metrics().xs("accuracy", level="metric")
    cand = s[s["candidate"].notna() & (s["metric"] == "accuracy")].set_index("candidate")
    n = int(cand["n"].iloc[0])
    rows = [f"| Candidate | Accuracy %, all 25,000 | Accuracy %, {n:,} without overlap | Difference, pp |",
            "|---|---:|---:|---:|"]
    for cid in protocol()["candidates"]:
        rows.append(f"| `{cid}` | {pct(m.loc[cid, 'point'], 2)} | {pct(cand.loc[cid, 'point'], 2)} | "
                    f"{pp(cand.loc[cid, 'point'] - m.loc[cid, 'point'])} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- docs/BENCHMARKS.md tables

def other_load() -> dict[str, pd.Series]:
    """Logical CPUs kept busy by other processes during each measurement, per benchmark stage."""
    return {"training runs": pd.read_csv(BENCH / "training_runs.csv")["other_busy_cpus"],
            "inference measurements": inference()["other_busy_cpus"],
            "HTTP levels": pd.read_csv(BENCH / "http.csv")["other_busy_cpus"]}


def table_environment() -> str:
    env = environment()
    hw, placement, stack = env["hardware"], env.get("placement", {}), env["http"].get("http_stack", {})
    cpus = placement.get("affinity_logical_cpus")
    rows = ["| Item | Value |", "|---|---|",
            f"| CPU | {hw.get('cpu', hw['machine'])}: {hw.get('physical_cores', '—')} cores, {hw['logical_cpus']} logical "
            f"CPUs; maximum clock reported by the operating system {hw.get('base_clock_mhz', '—'):,} MHz |",
            f"| Memory, operating system | {hw.get('ram_gb', '—')} GB; {hw['system']} |",
            f"| Software | CPython {hw['python']}, NumPy {hw['numpy']}, SciPy {hw['scipy']}, scikit-learn "
            f"{hw['scikit_learn']}, FastAPI {hw['fastapi']}, uvicorn {hw['uvicorn']} |",
            "| HTTP stack | " + ", ".join(f"{k} {v}" for k, v in stack.items() if v) + "; not installed: "
            + ", ".join(k for k, v in stack.items() if not v) + " |",
            "| Threads | " + ", ".join(f"`{k}={v}`" for k, v in env["thread_settings"].items()) + " |"]
    if cpus:
        rows.append(f"| Placement | Benchmark processes restricted to the {len(cpus)} logical CPUs of the performance "
                    f"cores ({', '.join(map(str, cpus))}) at above-normal priority |")
    for stage, values in other_load().items():
        rows.append(f"| Logical CPUs busy with other processes, {stage} | median {values.median():.1f} "
                    f"(range {values.min():.1f}–{values.max():.1f}) |")
    return "\n".join(rows)


def table_training() -> str:
    fits = training()
    rows = ["| Configuration | Training reviews | Features | Runs | Pipeline fit, s: median [range] "
            "| Cleaning / TF-IDF / estimator, s | Fit CPU, s | Peak working set, MB | Fit increment, MB "
            "| Max score difference vs frozen artefact | Other processes, busy CPUs |",
            "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for cid in trained():
        r = fits.loc[cid]
        rng = f"{sig(r['fit_s'])} [{sig(r['fit_s_min'])}–{sig(r['fit_s_max'])}]" if r["runs"] > 1 else sig(r["fit_s"])
        rows.append(f"| `{cid}` | {int(r['n_train']):,} | {int(r['n_features']):,} | {int(r['runs'])} | {rng} "
                    f"| {sig(r['cleaning_s'], 2)} / {sig(r['vectoriser_s'], 2)} / {sig(r['estimator_s'])} "
                    f"| {sig(r['fit_cpu_s'])} | {r['peak_mb']:,.0f} | {r['increment_mb']:,.0f} | {r['score_gap']:.1e} "
                    f"| {r['other_cpus']:.1f} |")
    return "\n".join(rows)


def table_inference() -> str:
    inf = inference()
    rows = ["| Configuration | Artefact, MB | Support vectors | Load, s | Single request p50 / p95 / p99, ms "
            "| Batch 64, predictions/s | Batch 1,024, predictions/s [range] | CPU s per 1,000: batch 64 / 1,024 "
            "[range at 1,024] |", "|---|---:|---:|---:|---|---:|---|---|"]
    for cid in trained():
        r = inf.loc[cid]
        support = f"{int(r['n_support_vectors']):,}" if r["n_support_vectors"] else "—"
        rows.append(f"| `{cid}` | {r['artefact_mb']:.1f} | {support} | {r['load_s_median']:.2f} "
                    f"| {r['latency_ms_p50']:.2f} / {r['latency_ms_p95']:.2f} / {r['latency_ms_p99']:.2f} "
                    f"| {r['throughput_b64_per_s']:,.0f} | {r['throughput_b1024_per_s']:,.0f} "
                    f"[{r['throughput_b1024_min']:,.0f}–{r['throughput_b1024_max']:,.0f}] "
                    f"| {sig(r['cpu_s_per_1000_b64'])} / {sig(r['cpu_s_per_1000_b1024'])} "
                    f"[{sig(r['cpu_s_per_1000_b1024_min'])}–{sig(r['cpu_s_per_1000_b1024_max'])}] |")
    return "\n".join(rows)


def table_http() -> str:
    levels = http_levels()
    rows = ["| Configuration | Concurrency | Throughput, requests/s [sweep range] | p50 / p95 / p99, ms "
            "| Failed / completed | Server CPU utilisation | Client CPU utilisation | Server CPU s per 1,000 requests |",
            "|---|---:|---|---|---|---:|---:|---:|"]
    for cid in protocol()["candidates"]:
        for r in levels[levels["candidate"] == cid].itertuples():
            rows.append(f"| `{cid}` | {r.concurrency} | {r.throughput:,.0f} [{r.throughput_min:,.0f}–{r.throughput_max:,.0f}] "
                        f"| {r.p50:.1f} / {r.p95:.1f} / {r.p99:.1f} | {int(r.failed)} / {int(r.completed):,} "
                        f"| {r.server_cpu_util:.2f} | {r.client_cpu_util:.2f} | {sig(r.server_cpu_s_per_1000)} |")
    return "\n".join(rows)


def table_tradeoffs() -> str:
    rows = ["| Comparison (A − B) | Factor | Δ accuracy, pp [95% CI] | Fit time, A / B [range] "
            "| Fit memory, A / B [range] | Inference CPU, A / B |", "|---|---|---|---|---|---:|"]
    for r in tradeoffs().itertuples():
        rows.append(f"| `{r.a}` − `{r.b}` | {r.factor} | {pp(r.d_acc)} {interval(r.d_lo, r.d_hi, pp)} "
                    f"| {ratio(r.time_ratio)} [{ratio(r.time_ratio_lo)}–{ratio(r.time_ratio_hi)}] "
                    f"| {ratio(r.memory_ratio)} [{ratio(r.memory_ratio_lo)}–{ratio(r.memory_ratio_hi)}] "
                    f"| {ratio(r.cpu_ratio)} |")
    return "\n".join(rows)


TABLES = {"summary": table_summary, "serving": table_serving, "economics": table_economics,
          "results": table_results, "paired": table_paired, "development": table_development,
          "ablation": table_ablation, "behavioural": table_behavioural, "overlap": table_overlap,
          "environment": table_environment, "training": table_training, "inference": table_inference,
          "http": table_http, "tradeoffs": table_tradeoffs}


def render(text: str) -> str:
    for name, make in TABLES.items():
        pattern = re.compile(rf"(<!-- table:{name} -->\n)(?:.*?\n)??(<!-- /table:{name} -->)", re.S)
        if pattern.search(text):
            body = make()
            text = pattern.sub(lambda match: match.group(1) + body + "\n" + match.group(2), text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path in DOCUMENTS:
        current = path.read_text(encoding="utf-8")
        updated = render(current)
        if updated == current:
            continue
        stale.append(path.relative_to(ROOT).as_posix())
        if not args.check:
            path.write_text(updated, encoding="utf-8", newline="\n")
    if args.check:
        if stale:
            print(f"generated tables are out of date in {', '.join(stale)}; run: python -m study.report")
            sys.exit(1)
        print("generated tables match the result files")
        return
    print(f"regenerated tables in {', '.join(stale)}" if stale else "generated tables already current")


if __name__ == "__main__":
    main()
