"""Figures for the official-test study, drawn from the saved result tables (never from new measurements).

    python -m study.figures

Reads results/official_test/{paired_differences,metrics}.csv and results/behavioural/failure_rates.csv and
writes to visual_outputs/, each in light and dark, wide (830 CSS px at 2x) and narrow (360 CSS px at 3x):
  09_paired_differences_*   paired accuracy differences with 95% and Bonferroni-adjusted intervals
  10_class_error_rates_*    class-conditional error rates per candidate with 95% intervals
Drawing primitives come from scripts/make_readme_figures.py, so all figures share one visual system.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_readme_figures import (BODY, MARK, NARROW, NARROW_SCALE, OUT, SMALL, THEMES, TITLE, WIDTH, Canvas,
                                         text_width, wrap)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "official_test"
PROTOCOL = ROOT / "protocol" / "protocol_v1.json"
ESTIMATOR = {"svc": "SVC", "linsvc": "LinearSVC", "logreg": "LogReg"}


def display_name(cid: str) -> str:
    if cid == "majority":
        return "Majority class"
    if cid == "svc_uni_8k_historical":
        return "SVC, unigrams, 8k (historical)"
    parts = cid.split("_")
    est, rest = ESTIMATOR[parts[0]], parts[1:]
    feats = "1-2-grams" if "bi" in rest else "unigrams"
    clean = ", v2" if "v2" in rest else ""
    size = "8k" if "8k" in rest else "25k"
    return f"{est}, {feats}{clean}, {size}"


def pp(v: float, digits: int = 2) -> str:
    return f"{v * 100:+.{digits}f}".replace("-", "−")


def nice_ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    step = next(s for s in (0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1) if span / s <= 7)
    return [v for v in np.arange(np.floor(lo / step) * step, hi + 1e-12, step) if lo - 1e-12 <= v <= hi + 1e-12]


def load_paired() -> pd.DataFrame:
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    df = pd.read_csv(RESULTS / "paired_differences.csv")
    df = df[(df["metric"] == "accuracy") & (df["block_ids"] == proto["statistics"]["block_ids"])]
    order = [c["id"] for c in proto["comparisons"]]
    return df.set_index("comparison").loc[order].reset_index()


def draw_paired(df: pd.DataFrame, theme: str, narrow: bool) -> Canvas:
    lo = min(0.0, df["lo_bonferroni"].min()) - 0.002
    hi = max(0.0, df["hi_bonferroni"].max()) + 0.002
    ticks = nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    title = "Paired accuracy differences on the official IMDB test partition"
    subtitle = ("Accuracy of A minus accuracy of B on the same 25,000 test reviews, in percentage points. Thick: 95% "
                "interval; thin: Bonferroni-adjusted for 7 comparisons. Stratified block bootstrap, 30-id blocks, "
                "10,000 replicates.")
    note = ("Above 0 favours A. Blocks of consecutive review ids approximate movie groups; movie identifiers are "
            "unavailable.")
    width, scale, m = (NARROW, NARROW_SCALE, 8) if narrow else (WIDTH, 2, 16)
    right = width - m
    if narrow:
        ax0, ax1 = m + 6, right - 6
        row_lines = [wrap(f"{display_name(r.a)} − {display_name(r.b)}", right - m, SMALL, "semibold") +
                     wrap(r.factor, right - m, SMALL) for r in df.itertuples()]
        heights = [10 + 16 * len(lines) + 30 for lines in row_lines]
    else:
        x_label, ax0, ax1, x_val = m, 372, 650, right
        row_lines = [wrap(f"{display_name(r.a)} − {display_name(r.b)}", ax0 - m - 20, SMALL, "semibold") +
                     wrap(r.factor, ax0 - m - 20, SMALL) for r in df.itertuples()]
        heights = [max(48, 14 + 16 * len(lines)) for lines in row_lines]

    def sx(v):
        return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

    t_lines = wrap(title, right - m, TITLE, "semibold")
    s_lines = wrap(subtitle, right - m, SMALL)
    n_lines = wrap(note, right - m, SMALL)
    top = 22 + 21 * (len(t_lines) - 1) + 20 + 16 * (len(s_lines) - 1) + (14 if narrow else 30)
    combined_gap = 8
    bottom = top + sum(heights) + combined_gap
    cv = Canvas(bottom + 44 + 16 * len(n_lines), theme, width=width, scale=scale)
    y = 22
    for line in t_lines:
        cv.text(m, y, line, size=TITLE, weight="semibold")
        y += 21
    y -= 1
    for line in s_lines:
        cv.text(m, y, line, size=SMALL, color="ink2")
        y += 16
    if not narrow:
        cv.text(m, top - 9, "Comparison (A − B) and the factor that differs", size=SMALL, color="ink2", weight="semibold")
        cv.text(ax0, top - 9, "Difference, percentage points", size=SMALL, color="ink2", weight="semibold")
        cv.text(x_val, top - 9, "A − B [95% interval]", size=SMALL, color="ink2", weight="semibold", ha="right")
    for v in ticks:
        cv.line([sx(v)] * 2, [bottom, bottom + 4], color="axis")
        cv.text(sx(v), bottom + 18, pp(v, 1) if v else "0", size=SMALL, color="ink2", ha="center")
    for i, line in enumerate(n_lines):
        cv.text(m, bottom + 40 + 16 * i, line, size=SMALL, color="ink2")

    y = top
    cv.line([m, right], [y, y])
    for k, (r, lines, h) in enumerate(zip(df.itertuples(), row_lines, heights)):
        if r.comparison.startswith("combined"):
            y += combined_gap
            cv.line([m, right], [y, y], color="axis")
        n_name = len(wrap(f"{display_name(r.a)} − {display_name(r.b)}", (right - m) if narrow else (ax0 - m - 20),
                          SMALL, "semibold"))
        for j, line in enumerate(lines):
            cv.text(m, y + 18 + 16 * j, line, size=SMALL, color="ink" if j < n_name else "ink2",
                    weight="semibold" if j < n_name else "normal")
        cy = (y + 18 + 16 * len(lines) + 8) if narrow else (y + h / 2)
        span = (cy - 9, cy + 9) if narrow else (y + 4, y + h - 4)
        label = f"{pp(r.point)} [{pp(r.lo95)}, {pp(r.hi95)}]"
        if narrow:  # the value label sits beside the interval, on whichever side has room
            right_side = sx(r.hi_bonferroni) + 6 + text_width(label, SMALL) <= right
            x0 = sx(r.hi_bonferroni) + 6 if right_side else sx(r.lo_bonferroni) - 6 - text_width(label, SMALL)
            clear = (x0 - 4, x0 + text_width(label, SMALL) + 4)
        else:
            clear = (np.inf, -np.inf)
        for v in ticks:
            if not clear[0] <= sx(v) <= clear[1]:
                cv.line([sx(v)] * 2, list(span), color="axis" if v == 0 else "grid")
        cv.line([sx(r.lo_bonferroni), sx(r.hi_bonferroni)], [cy, cy], color="ink2", width=1, z=2)
        cv.line([sx(r.lo95), sx(r.hi95)], [cy, cy], color="ink", width=3.5, z=3, cap="round")
        cv.ax.scatter([sx(r.point)], [cy], s=(10 * 0.75) ** 2, marker="o", color=cv.c["ink"],
                      edgecolors=cv.c["surface"], linewidths=1.5, zorder=4)
        if narrow:
            cv.text(x0, cy + 4, label, size=SMALL)
        else:
            cv.text(x_val, cy + 4.5, label, size=SMALL, ha="right")
        y += h
        cv.line([m, right], [y, y])
    return cv


def draw_class_errors(theme: str, narrow: bool) -> Canvas:
    proto = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    df = pd.read_csv(RESULTS / "metrics.csv")
    df = df[df["block_ids"] == proto["statistics"]["block_ids"]].set_index(["candidate", "metric"])
    cands = [c for c in proto["candidates"] if c != "majority"]
    lo = 0.0
    hi = max(df.loc[(c, f"error_{k}"), "hi95"] for c in cands for k in ("negative", "positive")) + 0.01
    hi = float(np.ceil(hi * 50) / 50)
    ticks = [v for v in np.arange(0, hi + 1e-9, 0.02 if hi <= 0.2 else 0.05)]
    width, scale, m = (NARROW, NARROW_SCALE, 8) if narrow else (WIDTH, 2, 16)
    right = width - m
    title = "Class-conditional error rates on the official IMDB test partition"
    subtitle = ("Share of each true class misclassified (12,500 reviews per class), with 95% stratified block-bootstrap "
                "intervals. The majority-class rule (100% / 0%) is omitted.")
    t_lines, s_lines = wrap(title, right - m, TITLE, "semibold"), wrap(subtitle, right - m, SMALL)
    legend_y = 22 + 21 * (len(t_lines) - 1) + 20 + 16 * (len(s_lines) - 1) + 24
    top = legend_y + (16 if narrow else 0) + 22  # the narrow legend takes two lines
    if narrow:
        ax0, ax1, row_h = m + 10, right - 16, 50
    else:
        ax0, ax1, row_h = 250, right - 150, 34
    bottom = top + row_h * len(cands)

    def sx(v):
        return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

    cv = Canvas(bottom + 30, theme, width=width, scale=scale)
    y = 22
    for line in t_lines:
        cv.text(m, y, line, size=TITLE, weight="semibold")
        y += 21
    y -= 1
    for line in s_lines:
        cv.text(m, y, line, size=SMALL, color="ink2")
        y += 16
    x = m
    for i, (cls, label) in enumerate((("negative", "negative reviews misclassified"),
                                      ("positive", "positive reviews misclassified"))):
        ly = legend_y + 16 * i if narrow else legend_y
        lx = m if narrow else x
        cv.marker(lx + 5, ly - 4.5, cls)
        cv.text(lx + 15, ly, label, size=SMALL, color="ink2")
        x += 15 + text_width(label, SMALL) + 20
    cv.text(right, legend_y, "error: negative / positive", size=SMALL, color="ink2", weight="semibold", ha="right")
    for v in ticks:
        cv.line([sx(v)] * 2, [bottom, bottom + 4], color="axis")
        cv.text(sx(v), bottom + 18, f"{v:.0%}", size=SMALL, color="ink2", ha="center")
    y = top
    cv.line([m, right], [y, y])
    for cid in cands:
        if narrow:
            cv.text(m, y + 16, display_name(cid), size=SMALL, weight="semibold")
            mid = [y + 30, y + 41]
        else:
            cv.text(m, y + row_h / 2 + 4.5, display_name(cid), size=SMALL, weight="semibold")
            mid = [y + row_h / 2 - 5, y + row_h / 2 + 5]
        span = (mid[0] - 6, mid[1] + 6)
        for v in ticks:
            cv.line([sx(v)] * 2, list(span), color="axis" if v == 0 else "grid")
        values = []
        for cls, cy in zip(("negative", "positive"), mid):
            row = df.loc[(cid, f"error_{cls}")]
            cv.line([sx(row["lo95"]), sx(row["hi95"])], [cy, cy], color=MARK[cls][1], width=2, z=2, cap="round")
            cv.marker(sx(row["point"]), cy, cls, size=9)
            values.append(f"{row['point']:.1%}")
        if not narrow:
            cv.text(right, y + row_h / 2 + 4.5, f"{values[0]}  /  {values[1]}", size=SMALL, ha="right")
        else:
            cv.text(right, y + 16, f"{values[0]} / {values[1]}", size=SMALL, ha="right")
        y += row_h
        cv.line([m, right], [y, y])
    return cv


def main() -> None:
    paired = load_paired()
    written = []
    for theme in THEMES:
        for narrow in (False, True):
            suffix = f"{'narrow_' if narrow else ''}{theme}"
            written.append(draw_paired(paired, theme, narrow).save(f"09_paired_differences_{suffix}.png"))
            written.append(draw_class_errors(theme, narrow).save(f"10_class_error_rates_{suffix}.png"))
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
