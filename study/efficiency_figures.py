"""Efficiency figures (11-13) and the provenance record of every study figure, drawn from committed result files.

    python -m study.efficiency_figures

Inputs: results/official_test/{metrics,paired_differences}.csv (frozen evaluation) and results/benchmarks/
{training_runs,inference,http}.csv (benchmarks on training-partition workloads). Outputs in visual_outputs/, each
in light and dark, wide (830 CSS px at 2x) and narrow (360 CSS px at 3x):
  11_accuracy_vs_compute_*     official-test accuracy against in-process CPU time per 1,000 predictions
  12_training_tradeoffs_*      fit-time ratio, fit-memory ratio and paired accuracy change per controlled comparison
  13_serving_latency_*         HTTP p50 and p95 latency against sustained throughput over a concurrency sweep
and visual_outputs/study_figures_provenance.json (SHA-256 of every input, the generating commit, and checks).
"""
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_readme_figures import (BODY, NARROW, NARROW_SCALE, OUT, SMALL, THEMES, TITLE, WIDTH, Canvas,
                                         text_width, wrap)
from study.figures import display_name, pp
from study.report import frontier, sig, tradeoffs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "protocol" / "protocol_v1.json"
SHAPE = {"svc": "s", "linear_svc": "o", "logreg": "^"}
ESTIMATOR_NAME = {"svc": "SVC", "linear_svc": "LinearSVC", "logreg": "LogReg"}
SHORT = {"size_svc": "Size, SVC (8k → 25k)", "size_linsvc": "Size, LinearSVC (8k → 25k)",
         "estimator_linsvc_vs_logreg": "LinearSVC vs LogReg", "estimator_svc_vs_linsvc": "SVC vs LinearSVC",
         "features_bigrams": "1–2-grams vs unigrams", "cleaning_v2": "Cleaning v2 vs legacy-v1",
         "combined_vs_historical": "Final vs historical (combined)"}
INPUTS = ["results/official_test/metrics.csv", "results/official_test/paired_differences.csv",
          "results/benchmarks/training_runs.csv", "results/benchmarks/inference.csv", "results/benchmarks/http.csv",
          "results/benchmarks/environment.json", "protocol/protocol_v1.json"]


def protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def log_scale(lo: float, hi: float, x0: float, x1: float):
    a, b = math.log10(lo), math.log10(hi)
    return lambda v: x0 + (math.log10(v) - a) / (b - a) * (x1 - x0)


def log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 3):
            v = m * 10 ** e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def fmt_tick(v: float, unit: str = "") -> str:
    text = f"{v:g}" if v >= 1 else f"{v:.2g}"
    return f"{text}{unit}"


def heading(cv: Canvas, m: int, right: int, title: str, subtitle: str) -> float:
    y = 22
    for line in wrap(title, right - m, TITLE, "semibold"):
        cv.text(m, y, line, size=TITLE, weight="semibold")
        y += 21
    y -= 1
    for line in wrap(subtitle, right - m, SMALL):
        cv.text(m, y, line, size=SMALL, color="ink2")
        y += 16
    return y


def heading_height(m: int, right: int, title: str, subtitle: str) -> float:
    return 22 + 21 * (len(wrap(title, right - m, TITLE, "semibold")) - 1) + 20 + 16 * (len(wrap(subtitle, right - m, SMALL)) - 1)


def place_label(text: str, px: float, py: float, obstacles: list[tuple[float, float, float, float]],
                bounds: tuple[float, float, float, float]) -> tuple[float, float, str]:
    """First candidate position (x, baseline y, ha) whose box avoids every obstacle box (x0, y0, x1, y1)."""
    w, h = text_width(text, SMALL), 12
    candidates = [(px + side * 9, py + dy, "left" if side > 0 else "right")
                  for dy in (4, -9, 17, -22, 30, -35, 43, -48) for side in (1, -1)]

    def box_of(x, y, ha):
        return (x, y - h + 2, x + w, y + 3) if ha == "left" else (x - w, y - h + 2, x, y + 3)

    def overlap(box):
        area = sum(max(0, min(box[2], o[2]) - max(box[0], o[0])) * max(0, min(box[3], o[3]) - max(box[1], o[1]))
                   for o in obstacles)
        outside = (max(0, bounds[0] - box[0]) + max(0, box[2] - bounds[2])) * h
        return area + outside

    scored = [(overlap(box_of(*c)), k, c) for k, c in enumerate(candidates)]
    best = min(scored)[2]  # the first candidate with no overlap, else the one with the least
    obstacles.append(box_of(*best))
    return best


def point(cv: Canvas, x: float, y: float, shape: str, filled: bool, color: str, size: float = 10, z: int = 5) -> None:
    area = (size * 0.75) ** 2 * (0.8 if shape == "s" else 1.0)
    face = cv.c[color] if filled else cv.c["surface"]
    cv.ax.scatter([x], [y], s=area, marker=shape, color=face, edgecolors=cv.c[color],
                  linewidths=1.6 if not filled else 1.2, zorder=z)


# ----------------------------------------------------------------------------- figure 11

def role_name(cid: str) -> str:
    """Display name with the configuration's role: the protocol's final model, the served default."""
    if cid == protocol()["final_model"]["candidate"]:
        return display_name(cid) + " (final model)"
    if cid == "svc_uni_8k_historical":
        return "SVC, unigrams, 8k (historical, served)"
    return display_name(cid)


def frontier_data() -> pd.DataFrame:
    return frontier()


def draw_frontier(df: pd.DataFrame, theme: str, narrow: bool) -> Canvas:
    width, scale, m = (NARROW, NARROW_SCALE, 8) if narrow else (WIDTH, 2, 16)
    right = width - m
    title = "Official-test accuracy versus inference compute"
    subtitle = ("Accuracy on the 25,000 official test reviews (95% block-bootstrap intervals) against CPU seconds per "
                "1,000 predictions, measured in-process on 2,000 training-partition reviews in batches of 1,024 on one "
                "thread (median and range of five passes). Dashed: Pareto-efficient set on point estimates, post hoc. "
                "Majority class (50.00%) omitted.")
    legend = [("s", True, "SVC"), ("o", True, "LinearSVC"), ("^", True, "LogReg"),
              ("o", True, "filled: 25,000 training reviews"), ("o", False, "hollow: 8,000")]
    top = heading_height(m, right, title, subtitle) + (76 if narrow else 54)  # legend row(s), then the axis title
    plot_h = 250 if narrow else 300
    x0, x1 = (m + 34, right - 8) if narrow else (m + 48, right - 150)
    y0, y1 = top + 8, top + plot_h
    acc_lo = math.floor((df["lo"].min() * 100 - 0.2) * 2) / 2
    acc_hi = math.ceil((df["hi"].max() * 100 + 0.2) * 2) / 2
    sy = lambda v: y1 - (v * 100 - acc_lo) / (acc_hi - acc_lo) * (y1 - y0)
    lo_x, hi_x = df["cpu_lo"].min() / 1.4, df["cpu_hi"].max() * 1.4
    sx = log_scale(lo_x, hi_x, x0, x1)
    key_lines = [f"{i + 1}  {role_name(r.candidate)}: {r.accuracy * 100:.2f}%, {sig(r.cpu_s_per_1000)} s"
                 for i, r in enumerate(df.itertuples())] if narrow else []
    height = y1 + 46 + (16 * len(key_lines) + 10 if narrow else 0)
    cv = Canvas(height, theme, width=width, scale=scale)
    y = heading(cv, m, right, title, subtitle)
    lx, ly = m, y + 12
    for shape, filled, label in legend:
        if narrow and lx + 18 + text_width(label, SMALL) > right:
            lx, ly = m, ly + 18
        point(cv, lx + 5, ly - 4, shape, filled, "ink2", size=8)
        cv.text(lx + 14, ly, label, size=SMALL, color="ink2")
        lx += 14 + text_width(label, SMALL) + 16
    for v in np.arange(acc_lo, acc_hi + 1e-9, 0.5 if narrow else 0.5):
        yy = sy(v / 100)
        cv.line([x0, x1], [yy, yy], color="grid")
        if abs(v - round(v)) < 1e-9:
            cv.text(x0 - 6, yy + 4, f"{v:.0f}%", size=SMALL, color="ink2", ha="right")
    for v in log_ticks(lo_x, hi_x):
        cv.line([sx(v)] * 2, [y0, y1], color="grid")
        cv.text(sx(v), y1 + 16, fmt_tick(v), size=SMALL, color="ink2", ha="center")
    cv.line([x0, x1], [y1, y1], color="axis")
    cv.text((x0 + x1) / 2, y1 + 34, "CPU seconds per 1,000 predictions (log scale)", size=SMALL, color="ink2", ha="center")
    cv.text(x0 - 6, y0 - 10, "Official-test accuracy", size=SMALL, color="ink2", ha="left")
    front = df[df["pareto"]]
    cv.ax.plot([sx(v) for v in front["cpu_s_per_1000"]], [sy(v) for v in front["accuracy"]], color=cv.c["ink2"],
               linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    obstacles = []  # markers and interval whiskers first, so labels avoid every point, not only earlier labels
    for r in df.itertuples():
        px = sx(r.cpu_s_per_1000)
        obstacles += [(px - 7, sy(r.accuracy) - 7, px + 7, sy(r.accuracy) + 7), (px - 2, sy(r.hi), px + 2, sy(r.lo)),
                      (sx(r.cpu_lo), sy(r.accuracy) - 2, sx(r.cpu_hi), sy(r.accuracy) + 2)]
    order = list(df.sort_values("pareto", ascending=False).index)  # efficient configurations claim space first
    for i in order:
        r = df.loc[i]
        px, py = sx(r.cpu_s_per_1000), sy(r.accuracy)
        color = "ink" if r.pareto else "ink2"
        cv.line([px, px], [sy(r.lo), sy(r.hi)], color=color, width=1.6, z=3)
        cv.line([sx(r.cpu_lo), sx(r.cpu_hi)], [py, py], color=color, width=1.6, z=3)
        point(cv, px, py, SHAPE[r.estimator], r.n_train > 10_000, color)
        label = str(i + 1) if narrow else role_name(r.candidate)
        lx, ly, ha = place_label(label, px, py, obstacles, (x0, y0, right, y1))
        cv.text(lx, ly, label, size=SMALL, color=color, weight="semibold" if r.pareto else "normal", ha=ha,
                knockout=True)
    if narrow:
        ky = y1 + 60
        for line in key_lines:
            cv.text(m, ky, line, size=SMALL, color="ink2")
            ky += 16
    return cv


# ----------------------------------------------------------------------------- figure 12

def tradeoff_data() -> pd.DataFrame:
    return tradeoffs()


def ratio_label(v: float) -> str:
    return f"×{sig(v, 3 if v >= 10 else 2)}"


def draw_tradeoffs(df: pd.DataFrame, theme: str, narrow: bool) -> Canvas:
    width, scale, m = (NARROW, NARROW_SCALE, 8) if narrow else (WIDTH, 2, 16)
    right = width - m
    title = "Training cost and accuracy change of each controlled comparison"
    subtitle = ("Configuration A relative to B. Fit time: complete Pipeline fit (cleaning, TF-IDF, estimator), median "
                "of three runs (one for the full-partition SVC), one thread; whiskers span all run pairs. Memory: peak "
                "working-set increase during the fit. Accuracy: paired official-test difference with 95% interval.")
    rlo = min(df["time_ratio_lo"].min(), df["memory_ratio_lo"].min(), 0.5) / 1.4
    rhi = max(df["time_ratio_hi"].max(), df["memory_ratio_hi"].max(), 2) * 1.6
    dlo = min(0, df["d_lo"].min()) - 0.004
    dhi = max(0, df["d_hi"].max()) + 0.004
    panels = [("Fit time, A / B (log)", "time_ratio", "log"), ("Fit memory, A / B (log)", "memory_ratio", "log"),
              ("Accuracy, A − B (pp)", "d_acc", "lin")]
    head = heading_height(m, right, title, subtitle)
    if narrow:
        row_h, label_w = 22, 0
        section_h = 34 + row_h * len(df) + 26
        height = head + 16 + section_h * 3 + 8
    else:
        row_h, label_w = 30, 220
        height = head + 44 + row_h * len(df) + 40
    cv = Canvas(height, theme, width=width, scale=scale)
    heading(cv, m, right, title, subtitle)
    if narrow:
        boxes = []
        y = head + 16
        label_col = max(text_width(SHORT[c], SMALL) for c in df["comparison"]) + 10
        for name, key, kind in panels:
            boxes.append((name, key, kind, y + 26, m + label_col, right - 44))
            y += section_h
    else:
        x_start = m + label_w + 12
        span = (right - x_start - 2 * 16) / 3
        boxes = [(name, key, kind, head + 44, x_start + i * (span + 16), x_start + i * (span + 16) + span - 46)
                 for i, (name, key, kind) in enumerate(panels)]
        for i, r in enumerate(df.itertuples()):
            cy = head + 44 + row_h * i + row_h / 2
            cv.text(m, cy + 4, SHORT[r.comparison], size=SMALL, weight="semibold" if r.comparison.startswith("combined") else "normal")
            cv.line([m, right], [cy + row_h / 2, cy + row_h / 2], color="grid")
    for name, key, kind, top, x0, x1 in boxes:
        if kind == "log":
            sx = log_scale(rlo, rhi, x0, x1)
            ticks = [v for v in (0.1, 0.3, 1, 3, 10, 30, 100, 300) if rlo <= v <= rhi]
            ref = 1.0
        else:
            sx = lambda v, x0=x0, x1=x1: x0 + (v - dlo) / (dhi - dlo) * (x1 - x0)
            step = 0.01 if dhi - dlo <= 0.06 else 0.02
            ticks = [v for v in np.arange(math.ceil(dlo / step) * step, dhi, step)]
            ref = 0.0
        bottom = top + row_h * len(df)
        cv.text(x0 if not narrow else m, top - 10, name, size=SMALL, color="ink2", weight="semibold")
        for v in ticks:
            cv.line([sx(v)] * 2, [top, bottom], color="axis" if abs(v - ref) < 1e-12 else "grid")
            if kind == "log" and abs(math.log10(v) - round(math.log10(v))) > 1e-9:
                continue  # unlabelled 3x gridlines: labels at every half decade collide in narrow panels
            cv.text(sx(v), bottom + 16, fmt_tick(v) if kind == "log" else (pp(v, 0) if v else "0"), size=SMALL,
                    color="ink2", ha="center")
        for i, r in enumerate(df.itertuples()):
            cy = top + row_h * i + row_h / 2
            if narrow:
                cv.text(m, cy + 4, SHORT[r.comparison], size=SMALL)
            if kind == "log":
                lo_, hi_ = getattr(r, f"{key}_lo"), getattr(r, f"{key}_hi")
                value, label = getattr(r, key), ratio_label(getattr(r, key))
            else:
                lo_, hi_, value, label = r.d_lo, r.d_hi, r.d_acc, pp(r.d_acc)
            cv.line([sx(lo_), sx(hi_)], [cy, cy], color="ink2", width=1.6, z=2, cap="round")
            point(cv, sx(value), cy, "o", True, "ink", size=8)
            cv.text(min(sx(hi_), x1) + 6, cy + 4, label, size=SMALL, color="ink", knockout=True)
    return cv


# ----------------------------------------------------------------------------- figure 13

def serving_data() -> pd.DataFrame:
    http = pd.read_csv(RESULTS / "benchmarks" / "http.csv")
    agg = (http.groupby(["candidate", "concurrency"])
           .agg(throughput=("throughput_rps", "median"), p50=("latency_ms_p50", "median"),
                p95=("latency_ms_p95", "median"), completed=("completed", "sum"), failed=("failed", "sum"))
           .reset_index())
    return agg


SERVE_STYLE = {"svc_uni_8k_historical": ("s", "ink2"), "linsvc_uni_full": ("o", "pos"), "linsvc_bi_v2_full": ("D", "ink")}


def draw_serving(df: pd.DataFrame, theme: str, narrow: bool) -> Canvas:
    width, scale, m = (NARROW, NARROW_SCALE, 8) if narrow else (WIDTH, 2, 16)
    right = width - m
    failed, total = int(df["failed"].sum()), int(df["completed"].sum())
    title = "HTTP serving latency versus sustained throughput"
    subtitle = ("One uvicorn worker, loopback HTTP, batch size 1; closed-loop clients at concurrency 1, 2, 4, 8, 16 "
                "and 32; 15 s per level after a 2 s warm-up; median of two sweeps; client on the same host. Solid: "
                f"p50 latency; dashed: p95. " + (f"None of the {total:,} requests failed." if not failed
                                                 else f"{failed:,} of {total:,} requests failed."))
    head = heading_height(m, right, title, subtitle)
    top = head + (40 if narrow else 30)
    plot_h = 240 if narrow else 300
    x0, x1 = (m + 40, right - 10) if narrow else (m + 52, right - 190)
    y0, y1 = top, top + plot_h
    tx_lo, tx_hi = df["throughput"].min() / 1.5, df["throughput"].max() * 1.5
    lat_lo, lat_hi = df["p50"].min() / 1.5, df["p95"].max() * 1.5
    sx = log_scale(tx_lo, tx_hi, x0, x1)
    ly = log_scale(lat_lo, lat_hi, 0, 1)
    sy = lambda v: y1 - ly(v) * (y1 - y0)
    names = {cid: role_name(cid) for cid in SERVE_STYLE}
    label_y, used = {}, []
    for cid in sorted(SERVE_STYLE, key=lambda c: -df[df["candidate"] == c]["p50"].iloc[-1]):  # top label first
        yv = sy(df[df["candidate"] == cid].sort_values("concurrency")["p50"].iloc[-1]) + 4
        for prev in used:
            yv = max(yv, prev + 34)
        used.append(yv)
        label_y[cid] = yv
    height = y1 + 44 + (18 * 3 + 6 if narrow else 0)
    cv = Canvas(height, theme, width=width, scale=scale)
    heading(cv, m, right, title, subtitle)
    for v in log_ticks(lat_lo, lat_hi):
        cv.line([x0, x1], [sy(v)] * 2, color="grid")
        cv.text(x0 - 6, sy(v) + 4, fmt_tick(v), size=SMALL, color="ink2", ha="right")
    for v in log_ticks(tx_lo, tx_hi):
        cv.line([sx(v)] * 2, [y0, y1], color="grid")
        cv.text(sx(v), y1 + 16, fmt_tick(v), size=SMALL, color="ink2", ha="center")
    cv.line([x0, x1], [y1, y1], color="axis")
    cv.text((x0 + x1) / 2, y1 + 34, "Sustained throughput, requests per second (log scale)", size=SMALL, color="ink2",
            ha="center")
    cv.text(x0 - 6, y0 - 8, "Latency, ms (log)", size=SMALL, color="ink2", ha="left")
    for k, (cid, (shape, color)) in enumerate(SERVE_STYLE.items()):
        part = df[df["candidate"] == cid].sort_values("concurrency")
        xs = [sx(v) for v in part["throughput"]]
        cv.ax.plot(xs, [sy(v) for v in part["p50"]], color=cv.c[color], linewidth=1.6, zorder=3)
        cv.ax.plot(xs, [sy(v) for v in part["p95"]], color=cv.c[color], linewidth=1.2, linestyle=(0, (3, 2)), zorder=3)
        for xv, p50, p95 in zip(xs, part["p50"], part["p95"]):
            point(cv, xv, sy(p50), shape, True, color, size=8)
            point(cv, xv, sy(p95), shape, False, color, size=7)
        last = part.iloc[-1]
        first = part.iloc[0]
        flip = cid == "linsvc_uni_full"  # its end points sit beside the final model's: label the other side
        cv.text(sx(first["throughput"]) + (7 if flip else -6), sy(first["p50"]) + 4, "c=1", size=SMALL, color="ink2",
                ha="left" if flip else "right")
        cv.text(sx(last["throughput"]) + (-7 if flip else 7), sy(last["p95"]) + 4, "c=32", size=SMALL, color="ink2",
                ha="right" if flip else "left")
        if narrow:
            yk = y1 + 56 + 18 * k
            point(cv, m + 5, yk - 4, shape, True, color, size=8)
            cv.text(m + 15, yk, names[cid], size=SMALL, color="ink")
        else:
            ty = label_y[cid]
            point(cv, x1 + 18, ty - 4, shape, True, color, size=8)
            for j, line in enumerate(wrap(names[cid], right - (x1 + 28), SMALL)):
                cv.text(x1 + 28, ty + 15 * j, line, size=SMALL, color="ink" if j == 0 else "ink2",
                        weight="semibold" if j == 0 else "normal")
    return cv


# ----------------------------------------------------------------------------- provenance and entry point

def sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".csv", ".json", ".py"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_provenance(written: list[Path]) -> None:
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sources = ["study/figures.py", "study/efficiency_figures.py", "study/report.py", "scripts/make_readme_figures.py",
               "results"]
    record = {"generated_by": ["study/figures.py", "study/efficiency_figures.py", "study/report.py"],
              "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "git": {"commit": git("rev-parse", "HEAD"), "sources": sources,
                      "sources_match_commit": not git("status", "--porcelain", "--", *sources)},
              "inputs_sha256": {p: sha256(ROOT / p) for p in INPUTS + ["results/behavioural/failure_rates.csv"]},
              "figures": {
                  "09_paired_differences": "results/official_test/paired_differences.csv",
                  "10_class_error_rates": "results/official_test/metrics.csv",
                  "11_accuracy_vs_compute": ["results/official_test/metrics.csv", "results/benchmarks/inference.csv"],
                  "12_training_tradeoffs": ["results/benchmarks/training_runs.csv",
                                            "results/official_test/paired_differences.csv"],
                  "13_serving_latency": "results/benchmarks/http.csv"},
              "files": sorted(p.name for p in written),
              "notes": ["Pareto membership in figure 11 is computed post hoc from official-test point estimates and "
                        "was not used to select any configuration.",
                        "Benchmarks use official training-partition reviews; the official test partition is read "
                        "only through the frozen prediction files."]}
    (OUT / "study_figures_provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from study.figures import main as draw_09_10

    draw_09_10()
    front, trade, serving = frontier_data(), tradeoff_data(), serving_data()
    written = []
    for theme in THEMES:
        for narrow in (False, True):
            suffix = f"{'narrow_' if narrow else ''}{theme}"
            written.append(draw_frontier(front, theme, narrow).save(f"11_accuracy_vs_compute_{suffix}.png"))
            written.append(draw_tradeoffs(trade, theme, narrow).save(f"12_training_tradeoffs_{suffix}.png"))
            written.append(draw_serving(serving, theme, narrow).save(f"13_serving_latency_{suffix}.png"))
    write_provenance(written + sorted(OUT.glob("09_*.png")) + sorted(OUT.glob("10_*.png")))
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    print("Pareto-efficient (post hoc):", ", ".join(front.loc[front["pareto"], "candidate"]))


if __name__ == "__main__":
    main()
