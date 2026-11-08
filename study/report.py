"""Generate the README's result tables from the committed result files, or check that they are current.

    python -m study.report           # rewrite every <!-- table:NAME --> ... <!-- /table:NAME --> block in README.md
    python -m study.report --check   # exit 1 if any block differs from what the result files produce (CI runs this)

Tables are produced only from files under results/ and protocol/, so the README cannot drift from the
measurements. Numbers outside these blocks are checked by tests/test_report.py where they are quoted.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "protocol" / "protocol_v1.json"


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}"


def pp(v: float) -> str:
    return f"{v * 100:+.2f}".replace("-", "−")


def interval(lo: float, hi: float, fmt=pct) -> str:
    return f"{fmt(lo)}–{fmt(hi)}" if fmt is pct else f"[{fmt(lo)}, {fmt(hi)}]"


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def describe(spec: dict) -> tuple[str, str, str]:
    cfg = spec["config"]
    if cfg is None:
        return "constant", "—", "—"
    feats = {(1, 1): "unigrams", (1, 2): "1–2-grams"}[tuple(cfg["ngram_range"])]
    est = {"svc": "SVC (linear kernel)", "linear_svc": "LinearSVC", "logreg": "LogisticRegression"}[cfg["estimator"]]
    return est, f"{feats}, {cfg['cleaning']}", f"{cfg['C']:g}"


def table_results() -> str:
    proto = protocol()
    m = pd.read_csv(RESULTS / "official_test" / "metrics.csv")
    m = m[m["block_ids"] == proto["statistics"]["block_ids"]].set_index(["candidate", "metric"])
    rows = ["| Candidate | Estimator | Features, cleaning | C | Training reviews | Accuracy % [95% CI] | Macro F1 % "
            "[95% CI] | Error, negative class % | Error, positive class % |",
            "|---|---|---|---:|---:|---|---|---|---|"]
    for cid, spec in proto["candidates"].items():
        est, feats, c = describe(spec)
        cell = lambda k: f"{pct(m.loc[(cid, k), 'point'])} [{interval(m.loc[(cid, k), 'lo95'], m.loc[(cid, k), 'hi95'])}]"
        rows.append(f"| `{cid}` | {est} | {feats} | {c} | {spec['n_train']:,} | {cell('accuracy')} | {cell('macro_f1')} "
                    f"| {cell('error_negative')} | {cell('error_positive')} |")
    return "\n".join(rows)


def table_paired() -> str:
    proto = protocol()
    d = pd.read_csv(RESULTS / "official_test" / "paired_differences.csv")
    d = d[d["metric"] == "accuracy"].set_index(["comparison", "block_ids"])
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
    proto = protocol()
    b = pd.read_csv(RESULTS / "behavioural" / "failure_rates.csv")
    b = b[(b["split"] == "heldout") & (b["ambiguity"] == "all")].set_index(["candidate", "phenomenon"])
    phenomena = ["negation", "contrast", "sarcasm", "intensity"]
    support = {p: b.xs(p, level="phenomenon").iloc[0] for p in phenomena}
    head = " | ".join(f"{p.capitalize()} ({int(support[p]['units'])} units, {int(support[p]['families'])} families)"
                      for p in phenomena)
    rows = [f"| Candidate | {head} |", "|---|" + "---|" * len(phenomena)]
    for cid in proto["candidates"]:
        cells = []
        for p in phenomena:
            r = b.loc[(cid, p)]
            cells.append(f"{pct(r['failure_rate'], 0)} ({int(r['failed_units'])}/{int(r['units'])}) "
                         f"[{pct(r['lo95'], 0)}–{pct(r['hi95'], 0)}]")
        rows.append(f"| `{cid}` | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def table_cost() -> str:
    proto = protocol()
    inf = pd.read_csv(RESULTS / "official_test" / "inference.csv").set_index("candidate")
    rows = ["| Candidate | Fit time, s | Features | Batch scoring of 25,000 reviews, s | Single request p50 / p99, ms |",
            "|---|---:|---:|---:|---|"]
    for cid, spec in proto["candidates"].items():
        if spec["config"] is None:
            continue
        fit = json.loads((RESULTS / "study" / "fits" / f"{cid}.json").read_text(encoding="utf-8"))
        r = inf.loc[cid]
        n_feat = fit.get("n_features")
        rows.append(f"| `{cid}` | {fit['fit_seconds']:.1f} | {n_feat:,} | {r['batch_seconds']:.1f} | "
                    f"{r['latency_ms_p50']:.2f} / {r['latency_ms_p99']:.2f} |" if n_feat else
                    f"| `{cid}` | {fit['fit_seconds']:.1f} | — | {r['batch_seconds']:.1f} | "
                    f"{r['latency_ms_p50']:.2f} / {r['latency_ms_p99']:.2f} |")
    return "\n".join(rows)


def table_overlap() -> str:
    proto = protocol()
    s = pd.read_csv(RESULTS / "official_test" / "overlap_sensitivity.csv")
    m = pd.read_csv(RESULTS / "official_test" / "metrics.csv")
    m = m[(m["block_ids"] == proto["statistics"]["block_ids"]) & (m["metric"] == "accuracy")].set_index("candidate")
    cand = s[s["candidate"].notna() & (s["metric"] == "accuracy")].set_index("candidate")
    n = int(cand["n"].iloc[0])
    rows = [f"| Candidate | Accuracy %, all 25,000 | Accuracy %, {n:,} without overlap | Difference, pp |",
            "|---|---:|---:|---:|"]
    for cid in proto["candidates"]:
        rows.append(f"| `{cid}` | {pct(m.loc[cid, 'point'], 2)} | {pct(cand.loc[cid, 'point'], 2)} | "
                    f"{pp(cand.loc[cid, 'point'] - m.loc[cid, 'point'])} |")
    return "\n".join(rows)


TABLES = {"results": table_results, "paired": table_paired, "development": table_development,
          "ablation": table_ablation, "behavioural": table_behavioural, "cost": table_cost, "overlap": table_overlap}


def render(text: str) -> str:
    for name, make in TABLES.items():
        pattern = re.compile(rf"(<!-- table:{name} -->\n).*?(\n<!-- /table:{name} -->)", re.S)
        if pattern.search(text):
            body = make()
            text = pattern.sub(lambda match: match.group(1) + body + match.group(2), text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = README.read_text(encoding="utf-8")
    updated = render(current)
    if args.check:
        if updated != current:
            print("README.md result tables are out of date; run: python -m study.report")
            sys.exit(1)
        print("README.md result tables match the result files")
        return
    README.write_text(updated, encoding="utf-8", newline="\n")
    print("README.md result tables regenerated")


if __name__ == "__main__":
    main()
