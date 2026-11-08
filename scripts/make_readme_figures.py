"""Render the historical-model figures (06-08), light and dark, from the published historical Pipeline.

Run from the repo root after `python train.py`:

    python scripts/make_readme_figures.py

These figures describe the historical model and its internal holdout (2026-11-06 protocol); the
official-test study has its own figures (study/figures.py). Reads the Pipeline artefact that
train.py publishes (models/serving.json), models/test_predictions.csv, the original run's
metrics (results/historical/metrics.json) and data/training_data.csv. Before drawing, it checks
that they agree: the seeded split rebuilt from the data must give the rows of
test_predictions.csv, the model must reproduce every saved prediction, the recomputed accuracy
and confusion matrix must equal the original metrics, and the decomposed score must add up to
the model's own score. Any mismatch stops the script. Writes to visual_outputs/:

    06_probe_outputs_{light,dark}.png        five hand-written probes: expected, predicted, signed decision score
    07_error_by_length_{light,dark}.png      held-out error rate by review length and true class
    08_score_contributions_{light,dark}.png  one decision score split into weight x TF-IDF terms and the intercept
    0[678]_*_narrow_{light,dark}.png         each figure again, laid out for phone-width README columns
    06_probe_outputs.csv, 07_error_by_length.csv, 08_score_contributions.csv   the numbers behind each figure
    figures_provenance.json                  commit, input checksums, model version, checks and library versions

The PNGs are laid out for GitHub's README column (830 CSS px wide on a desktop browser) at 2x
pixel density, on GitHub's light (#ffffff) and dark (#0d1117) page colours. The class colours
are make_visuals.py's diverging poles (light) and their dark-mode steps; both pairs pass the
dataviz palette checks on these two surfaces. Needs matplotlib (dev-only, not in requirements.txt).
"""
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import BoxStyle, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # joblib must import src.preprocessor to unpickle it
from src.pipeline import LABELS, labels_from_scores, load  # noqa: E402
from train import wilson_interval  # noqa: E402

MODELS = ROOT / "models"
HISTORICAL_METRICS = ROOT / "results" / "historical" / "metrics.json"  # the original 2026-11-07 run
HISTORICAL_ID = "historical-svc-c1-uni-legacy-v1-8k"
OUT = ROOT / "visual_outputs"

WIDTH = 830  # CSS px: the README column on github.com in a desktop browser
SCALE = 2  # device px per CSS px, so text stays sharp on high-density screens
DPI = 96 * SCALE  # 1 CSS px = 1/96 in, so 10 pt text is 13.3 CSS px
TITLE, BODY, SMALL = 13, 10, 9  # pt: 17.3, 13.3 and 12 CSS px
# Each figure also comes in a narrow layout. The README shows it below a 960 px viewport:
# GitHub's column is 254-364 px on phones and 406-538 px on 768-959 px screens, where its sidebar returns.
NARROW = 360  # CSS px; the README caps it at this width
NARROW_SCALE = 3  # device px per CSS px; most phones have 3x screens
# Segoe UI matches make_visuals.py and covers every glyph used here; DejaVu Sans ships with matplotlib.
FONT = ["Segoe UI" if "Segoe UI" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Sans"]

THEMES = {
    "light": {"surface": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "grid": "#d1d9e0", "axis": "#818b98",
              "band": "#f3f5f7", "neg": "#e34948", "pos": "#2a78d6", "total": "#59636e"},
    "dark": {"surface": "#0d1117", "ink": "#e6edf3", "ink2": "#9198a1", "grid": "#30363d", "axis": "#656c76",
             "band": "#161b22", "neg": "#e66767", "pos": "#3987e5", "total": "#9198a1"},
}
MARK = {"negative": ("s", "neg"), "positive": ("o", "pos")}  # shape repeats colour: square = negative

# Hand-written probes, not dataset reviews. Probes 1, 2, 3 and 5 are the /docs screenshot
# examples (scripts/docs_screenshot.py); probe 4 is probe 5 without "not". "Expected" is the
# sentiment each sentence was written to express.
PROBES = [
    ("positive", "An absolute masterpiece: superb acting and a story I loved from start to finish."),
    ("negative", "Boring, predictable and badly acted. A total waste of two hours."),
    ("positive", "I expected to hate it, but it turned out to be surprisingly good."),
    ("positive", "The movie was good."),
    ("negative", "The movie was not good."),
]
EXPLAINED = 4  # the probe whose score figure 08 decomposes
TOP_TERMS = 8  # figure 08 lists at most this many terms and sums the rest as "other terms"
QUARTERS = ["shortest quarter", "second quarter", "third quarter", "longest quarter"]


# ----------------------------------------------------------------------------- evidence

def load_and_check():
    """Load the artefacts and stop unless they reproduce the reported evaluation exactly."""
    metrics = json.loads(HISTORICAL_METRICS.read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / Path(metrics["data"].replace("\\", "/")))
    held = pd.read_csv(MODELS / "test_predictions.csv")
    _, x_test, _, y_test = train_test_split(data["text"], data["label"], test_size=metrics["test_size"],
                                            stratify=data["label"], random_state=metrics["seed"])
    if x_test.tolist() != held["text"].tolist() or y_test.tolist() != held["label"].tolist():
        raise SystemExit("models/test_predictions.csv is not the seeded held-out split of the data; rerun train.py")

    pipeline, manifest = load(MODELS)
    if manifest["candidate_id"] != HISTORICAL_ID:
        raise SystemExit("models/serving.json does not point at the historical model; run: python train.py")
    clf = pipeline.named_steps["clf"]
    if list(clf.classes_) != [0, 1]:  # a positive score must mean classes_[1] = positive
        raise SystemExit(f"unexpected classes {clf.classes_}")
    scores = clf.decision_function(pipeline[:-1].transform(held["text"].tolist()))
    labels = labels_from_scores(scores)
    if not np.array_equal(labels, held["predicted"].to_numpy()):
        raise SystemExit("the model does not reproduce models/test_predictions.csv; rerun train.py")
    cm = confusion_matrix(held["label"], labels, labels=sorted(LABELS)).tolist()
    accuracy = float(np.mean(labels == held["label"].to_numpy()))
    if cm != metrics["confusion_matrix"]["rows_true_cols_predicted"] or not math.isclose(accuracy, metrics["accuracy"]):
        raise SystemExit("the recomputed confusion matrix or accuracy differs from results/historical/metrics.json")
    held["score"] = scores
    held["words"] = held["text"].str.split().str.len()
    return metrics, pipeline, manifest, held


def score_probes(pipeline):
    scores = pipeline.decision_function([text for _, text in PROBES])
    labels = labels_from_scores(scores)
    rows = []
    for n, ((expected, text), label, score) in enumerate(zip(PROBES, labels, scores), 1):
        predicted = LABELS[int(label)]
        rows.append({"probe": n, "text": text, "expected": expected, "predicted": predicted,
                     "decision_score": float(score), "correct": predicted == expected})
    return pd.DataFrame(rows)


def error_by_length(held):
    """Error rate per true class within quartiles of word count, pooled over both classes."""
    quarter = pd.qcut(held["words"], 4, labels=False)
    groups = [(QUARTERS[q], quarter == q) for q in range(4)] + [("all lengths", held["words"] > 0)]
    rows = []
    for name, in_group in groups:
        words = held.loc[in_group, "words"]
        for label in sorted(LABELS):
            g = held[in_group & (held["label"] == label)]
            errors, reviews = int((g["predicted"] != g["label"]).sum()), len(g)
            low, high = wilson_interval(errors, reviews)
            rows.append({"length_group": name, "min_words": int(words.min()), "max_words": int(words.max()),
                         "true_class": LABELS[label], "errors": errors, "reviews": reviews,
                         "error_rate": errors / reviews, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def contributions(pipeline, text):
    """Split one decision score into intercept + weight x TF-IDF per term, and check the sum."""
    clf = pipeline.named_steps["clf"]
    x = pipeline[:-1].transform([text])
    coef = clf.coef_
    coef = coef.toarray().ravel() if hasattr(coef, "toarray") else np.ravel(coef)
    terms = pipeline.named_steps["tfidf"].get_feature_names_out()
    intercept = float(clf.intercept_[0])
    score = float(clf.decision_function(x)[0])
    parts = sorted(((terms[j], float(v), float(coef[j])) for j, v in zip(x.indices, x.data)),
                   key=lambda p: -abs(p[1] * p[2]))
    shown, rest = parts[:TOP_TERMS], parts[TOP_TERMS:]
    rows = [{"component": term, "kind": "term", "tfidf": v, "weight": w, "contribution": v * w}
            for term, v, w in shown]
    rows.append({"component": f"other terms ({len(rest)})", "kind": "other", "tfidf": None, "weight": None,
                 "contribution": float(sum(v * w for _, v, w in rest))})
    rows.append({"component": "intercept", "kind": "intercept", "tfidf": None, "weight": None,
                 "contribution": intercept})
    table = pd.DataFrame(rows)
    table["running_total"] = table["contribution"].cumsum()
    total = float(table["running_total"].iloc[-1])
    gap = abs(total - score)
    if gap > 1e-9:
        raise SystemExit(f"contributions sum to {total!r}, but the model scores {score!r}")
    table = pd.concat([table, pd.DataFrame([{"component": "signed decision score", "kind": "total", "tfidf": None,
                                             "weight": None, "contribution": score, "running_total": score}])],
                      ignore_index=True)
    return table, len(parts), gap


# ----------------------------------------------------------------------------- drawing

_MEASURE = plt.figure(dpi=DPI)
_RENDERER = _MEASURE.canvas.get_renderer()


def text_width(s, size, weight="normal"):
    """Rendered width of a string in CSS px."""
    prop = FontProperties(family=FONT, size=size, weight=weight)
    return _RENDERER.get_text_width_height_descent(s, prop, ismath=False)[0] / SCALE


def wrap(s, width, size, weight="normal"):
    """Greedy word wrap by measured width, so no line overflows its column."""
    lines, line = [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if line and text_width(trial, size, weight) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def signed(v, digits=2):
    """+1.12 / −3.86, with a true minus sign."""
    return f"{v:+.{digits}f}".replace("-", "−")


class Canvas:
    """One figure in CSS px with the origin at the top left, so layout reads like a web page."""

    def __init__(self, height, theme, width=WIDTH, scale=SCALE):
        self.c = THEMES[theme]
        self.width, self.height, self.dpi = width, height, 96 * scale
        self.fig = plt.figure(figsize=(width / 96, height / 96), dpi=self.dpi)
        self.fig.patch.set_facecolor(self.c["surface"])
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_axis_off()

    def text(self, x, y, s, size=BODY, color="ink", weight="normal", ha="left", z=5, knockout=False):
        """Text whose baseline sits at y; `knockout` clears the lines behind it with the surface colour,
        or with the named theme colour (such as "band") when it is a string."""
        fill = self.c[knockout] if isinstance(knockout, str) else self.c["surface"]
        box = {"facecolor": fill, "edgecolor": "none", "pad": 1.5} if knockout else None
        self.ax.text(x, y, s, fontsize=size, color=self.c[color], fontweight=weight, family=FONT, ha=ha,
                     va="baseline", zorder=z, parse_math=False, bbox=box)

    def heading(self, title, subtitle):
        """Question-led title and a wrapped subtitle; returns the y below them."""
        self.text(16, 27, title, size=TITLE, weight="semibold")
        y = 50
        for line in wrap(subtitle, self.width - 32, SMALL):
            self.text(16, y, line, size=SMALL, color="ink2")
            y += 17
        return y

    def line(self, xs, ys, color="grid", width=1.0, z=1, cap="butt"):
        self.ax.plot(xs, ys, color=self.c[color], linewidth=width * 0.75, solid_capstyle=cap, zorder=z)

    def band(self, x, y, w, h, color="band"):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=self.c[color], linewidth=0, zorder=0))

    def marker(self, x, y, cls, size=10, z=4):
        """Class marker with a 2 px ring in the surface colour; size is its width in CSS px."""
        shape, color = MARK[cls]
        area = (size * 0.75) ** 2 * (0.8 if shape == "s" else 1.0)  # equal visual weight for both shapes
        self.ax.scatter([x], [y], s=area, marker=shape, color=self.c[color], edgecolors=self.c["surface"],
                        linewidths=1.5, zorder=z)

    def bar(self, x0, x1, y, h, color, square_at=None, radius=3, z=3):
        """Horizontal bar; rounded ends, except a square end at `square_at` (a baseline)."""
        lo, hi = sorted((x0, x1))
        width = hi - lo
        if width < 2 * radius:  # too short to round; keep it visible as a 2 px stub
            centre = (lo + hi) / 2
            lo, width = centre - max(width, 2) / 2, max(width, 2)
            self.ax.add_patch(Rectangle((lo, y - h / 2), width, h, facecolor=self.c[color], linewidth=0, zorder=z))
            return
        self.ax.add_patch(FancyBboxPatch((lo, y - h / 2), width, h, facecolor=self.c[color], linewidth=0, zorder=z,
                                         boxstyle=BoxStyle.Round(pad=0, rounding_size=radius)))
        if square_at is not None:
            near = lo if abs(square_at - lo) < abs(square_at - hi) else hi - radius
            self.ax.add_patch(Rectangle((near, y - h / 2), radius, h, facecolor=self.c[color], linewidth=0,
                                        zorder=z))

    def save(self, name):
        path = OUT / name
        self.fig.savefig(path, dpi=self.dpi, facecolor=self.c["surface"], metadata={"Software": None})
        plt.close(self.fig)
        return path


def draw_probes(probes, version, theme):
    x_in, w_in, x_exp, x_pred = 16, 300, 334, 432
    ax0, ax1, lo, hi = 562, 806, -4.3, 2.0
    line_h, pad = 18, 9

    def sx(v):
        return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

    texts = [wrap(f"“{t}”", w_in, BODY) for t in probes["text"]]
    heights = [max(len(lines), 2) * line_h + 2 * pad for lines in texts]
    top = 92
    bottom = top + sum(heights)
    cv = Canvas(bottom + 50, theme)
    cv.heading("Signed decision scores of the historical model on five constructed probes",
               f"Historical model {version} (SVC, legacy-v1 unigrams, 8,000 training reviews). Expected is the "
               "sentiment each sentence was written to express; it is not a dataset label.")
    for x, label in ((x_in, "Input"), (x_exp, "Expected"), (x_pred, "Predicted"), (ax0, "Signed decision score")):
        cv.text(x, top - 10, label, size=SMALL, color="ink2", weight="semibold")
    cv.line([sx(0)] * 2, [top, bottom], color="axis")  # the decision boundary; value labels carry the rest
    for v in (-4, -2, 0, 2):
        cv.line([sx(v)] * 2, [bottom, bottom + 4], color="axis")
        cv.text(sx(v), bottom + 18, signed(v, 0) if v else "0", size=SMALL, color="ink2", ha="center")
    cv.text(sx(0) - 7, bottom + 36, "← negative", size=SMALL, color="ink2", ha="right")
    cv.text(sx(0) + 7, bottom + 36, "positive →", size=SMALL, color="ink2")
    cv.text(16, bottom + 36, "Label rule: above 0 is positive; 0 or below is negative. Not a probability.",
            size=SMALL, color="ink2")

    y = top
    cv.line([16, WIDTH - 16], [y, y])
    for lines, h, row in zip(texts, heights, probes.itertuples()):
        if not row.correct:
            cv.band(16, y, WIDTH - 32, h)
        base = y + pad + 13  # baseline of the first text line
        mid = base - 4.5  # visual centre of that line
        for i, text in enumerate(lines):
            cv.text(x_in, base + i * line_h, text)
        cv.marker(x_exp + 5, mid, row.expected)
        cv.text(x_exp + 15, base, row.expected)
        cv.marker(x_pred + 5, mid, row.predicted)
        cv.text(x_pred + 15, base, row.predicted)
        if row.correct:
            cv.text(x_pred + 15, base + line_h, "correct", size=SMALL, color="ink2")
        else:
            cv.text(x_pred + 15, base + line_h, "wrong", weight="semibold")
        cv.marker(sx(row.decision_score), mid, row.predicted, size=11)
        cv.text(sx(row.decision_score) + 10, base, signed(row.decision_score), size=SMALL)
        y += h
        cv.line([16, WIDTH - 16], [y, y])
    return cv


def draw_probes_narrow(probes, version, theme):
    """Figure 06 for a phone-width column: each probe's text, then its labels, then its score on a strip."""
    m, line_h, pad = 8, 18, 8
    right = NARROW - m
    lo, hi = -4.3, 2.0  # the wide figure's score range
    ax0, ax1 = m + 6, right - 16  # +2 sits far enough right that +1.12's value label clears its tick

    def sx(v):
        return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)

    title = wrap("Signed decision scores of the historical model on five constructed probes", right - m, TITLE, "semibold")
    subtitle = wrap(f"Historical model {version}. Expected is what each sentence was written to "
                    "express, not a dataset label.", right - m, SMALL)
    note = [line for sentence in ("Label rule: above 0 is positive; 0 or below is negative.",
                                  "The score is not a probability.") for line in wrap(sentence, right - m, SMALL)]
    texts = [wrap(f"“{t}”", right - m, BODY) for t in probes["text"]]
    heights = [pad + 13 + (len(lines) - 1) * line_h + 20 + 17 + 8 + pad for lines in texts]
    top = 22 + 21 * (len(title) - 1) + 20 + 16 * (len(subtitle) - 1) + 14
    bottom = top + sum(heights)
    cv = Canvas(bottom + 56 + 16 * (len(note) - 1) + 12, theme, width=NARROW, scale=NARROW_SCALE)

    y = 22
    for line in title:
        cv.text(m, y, line, size=TITLE, weight="semibold")
        y += 21
    y -= 1  # the subtitle starts 20 px below the title's last baseline
    for line in subtitle:
        cv.text(m, y, line, size=SMALL, color="ink2")
        y += 16

    y = top
    cv.line([m, right], [y, y])
    for lines, h, row in zip(texts, heights, probes.itertuples()):
        if not row.correct:
            cv.band(m, y, right - m, h)
        base = y + pad + 13
        for i, line in enumerate(lines):
            cv.text(m, base + i * line_h, line)
        labels = base + (len(lines) - 1) * line_h + 20
        x = m
        for caption, cls in (("expected", row.expected), ("predicted", row.predicted)):
            cv.text(x, labels, caption, size=SMALL, color="ink2")
            x += text_width(caption, SMALL) + 7
            cv.marker(x + 4.5, labels - 4, cls, size=9)
            x += 13
            cv.text(x, labels, cls, size=SMALL)
            x += text_width(cls, SMALL) + 16
        if row.correct:
            cv.text(right, labels, "correct", size=SMALL, color="ink2", ha="right")
        else:
            cv.text(right, labels, "wrong", size=SMALL, weight="semibold", ha="right")
        mid = labels + 17  # centre of this probe's score strip
        cv.line([ax0, ax1], [mid, mid])
        for v in (-4, -2, 2):
            cv.line([sx(v)] * 2, [mid - 2.5, mid + 2.5])
        cv.line([sx(0)] * 2, [mid - 7, mid + 7], color="axis")  # the decision boundary
        cv.marker(sx(row.decision_score), mid, row.predicted, size=10)
        cv.text(sx(row.decision_score) + 9, mid + 4, signed(row.decision_score), size=SMALL,
                knockout=True if row.correct else "band")  # clear the track behind the value
        y += h
        cv.line([m, right], [y, y])

    for v in (-4, -2, 0, 2):
        cv.line([sx(v)] * 2, [bottom, bottom + 4], color="axis")
        cv.text(sx(v), bottom + 18, signed(v, 0) if v else "0", size=SMALL, color="ink2", ha="center")
    cv.text(sx(0) - 7, bottom + 34, "← negative", size=SMALL, color="ink2", ha="right")
    cv.text(sx(0) + 7, bottom + 34, "positive →", size=SMALL, color="ink2")
    y = bottom + 56
    for line in note:
        cv.text(m, y, line, size=SMALL, color="ink2")
        y += 16
    return cv


def draw_length(table, theme):
    x_label, px0, px1, x_rate, x_count = 16, 196, 592, 668, WIDTH - 16
    xmax, row_h = 0.25, 48

    def sx(v):
        return px0 + v / xmax * (px1 - px0)

    groups = list(dict.fromkeys(table["length_group"]))
    subtitle = ("Historical model on its internal holdout (2,000 training-partition reviews). Error rate by true class "
                "within each length quartile: the negative-class rate is higher in all four; neither rate shows a clear "
                "trend with length.")
    head_lines = len(wrap(subtitle, WIDTH - 32, SMALL))
    top = 50 + 17 * head_lines + 46
    bottom = top + row_h * len(groups) + 8
    footnote = wrap("Exploratory. Quarters split all 2,000 held-out reviews by word count. Each 95% Wilson interval "
                    "covers one cell, treats reviews as independent and is not adjusted for multiple comparisons.",
                    WIDTH - 32, SMALL)
    cv = Canvas(bottom + 52 + 17 * (len(footnote) - 1), theme)
    cv.heading("Class-conditional error rates across review-length quartiles", subtitle)
    legend_y = 50 + 17 * head_lines + 8
    x = 16
    for cls, label in (("negative", "Negative reviews (true label), wrongly called positive"),
                       ("positive", "Positive reviews (true label), wrongly called negative")):
        cv.marker(x + 5, legend_y - 4.5, cls)
        cv.text(x + 15, legend_y, label, size=SMALL, color="ink2")
        x += 15 + text_width(label, SMALL) + 24
    head_y = top - 9
    cv.text(x_label, head_y, "Review length", size=SMALL, color="ink2", weight="semibold")
    cv.text(px0, head_y, "Error rate, 95% Wilson interval", size=SMALL, color="ink2", weight="semibold")
    cv.text(x_rate, head_y, "Rate", size=SMALL, color="ink2", weight="semibold", ha="right")
    cv.text(x_count, head_y, "Errors / reviews", size=SMALL, color="ink2", weight="semibold", ha="right")
    for v in np.arange(0, xmax + 1e-9, 0.05):
        cv.line([sx(v)] * 2, [top, bottom], color="axis" if v == 0 else "grid")
        cv.text(sx(v), bottom + 17, f"{v:.0%}", size=SMALL, color="ink2", ha="center")
    for i, line in enumerate(footnote):
        cv.text(16, bottom + 38 + 17 * i, line, size=SMALL, color="ink2")

    y = top
    cv.line([16, WIDTH - 16], [y, y])
    for i, name in enumerate(groups):
        if name == "all lengths":
            y += 8
            cv.line([16, WIDTH - 16], [y, y], color="axis")
        rows = table[table["length_group"] == name]
        first = rows.iloc[0]
        span = f"{first.min_words:,}–{first.max_words:,} words"
        cv.text(x_label, y + 20, "All lengths" if name == "all lengths" else span, weight="semibold" if name == "all lengths" else "normal")
        cv.text(x_label, y + 37, f"{span}, {int(rows.reviews.sum()):,} reviews" if name == "all lengths" else
                f"{name}, {int(rows.reviews.sum()):,} reviews", size=SMALL, color="ink2")
        for j, r in enumerate(rows.itertuples()):
            cy = y + 15 + 18 * j
            color = MARK[r.true_class][1]
            cv.line([sx(r.ci95_low), sx(r.ci95_high)], [cy, cy], color=color, width=2, z=2, cap="round")
            cv.marker(sx(r.error_rate), cy, r.true_class)
            cv.text(x_rate, cy + 4.5, f"{r.error_rate:.1%}", size=SMALL, ha="right")
            cv.text(x_count, cy + 4.5, f"{r.errors:,} / {r.reviews:,}", size=SMALL, ha="right")
        y += row_h
        if name != "all lengths":
            cv.line([16, WIDTH - 16], [y, y])
    return cv


def draw_contributions(table, text, n_terms, theme):
    x_term, x_calc, px0, px1, x_val = 16, 140, 312, 704, WIDTH - 16
    row_h, bar_h = 30, 14
    run = [0.0] + table["running_total"].tolist()[:-1]
    lo = min(0.0, min(run)) - 0.05
    hi = max(run) + 0.05

    def sx(v):
        return px0 + (v - lo) / (hi - lo) * (px1 - px0)

    subtitle = ("The score is the intercept plus, for each word of the text in the vocabulary, its SVM weight times "
                "its TF-IDF value. Bars add those terms in order of size. These are contributions within this "
                "linear model, not causal explanations of language.")
    head_lines = len(wrap(subtitle, WIDTH - 32, SMALL))
    top = 50 + 17 * head_lines + 30
    bottom = top + row_h * len(table) + 8
    cv = Canvas(bottom + 34, theme)
    cv.heading(f"Additive decomposition of the linear decision function: “{text}”", subtitle)
    head_y = top - 9
    cv.text(x_term, head_y, "Term", size=SMALL, color="ink2", weight="semibold")
    cv.text(x_calc, head_y, "Weight × TF-IDF", size=SMALL, color="ink2", weight="semibold")
    cv.text(px0, head_y, "Running total of the signed decision score", size=SMALL, color="ink2", weight="semibold")
    cv.text(x_val, head_y, "Contribution", size=SMALL, color="ink2", weight="semibold", ha="right")
    ticks = np.arange(0, hi, 0.25)
    for v in ticks:
        cv.line([sx(v)] * 2, [top, bottom], color="axis" if v == 0 else "grid")
        cv.text(sx(v), bottom + 17, signed(v) if v else "0", size=SMALL, color="ink2", ha="center")

    y = top
    cv.line([16, WIDTH - 16], [y, y])
    for i, r in enumerate(table.itertuples()):
        if r.kind == "total":
            y += 8
            cv.line([16, WIDTH - 16], [y, y], color="axis")
        cy = y + row_h / 2
        base = cy + 4.5
        if r.kind == "term":
            cv.text(x_term, base, r.component, weight="semibold")
            # four decimals, so the shown factors multiply back to the shown contribution
            cv.text(x_calc, base, f"{signed(r.weight, 4)} × {r.tfidf:.4f}", size=SMALL, color="ink2")
        elif r.kind == "other":
            cv.text(x_term, base, "other terms")
            cv.text(x_calc, base, f"none: all {n_terms} terms shown" if n_terms <= TOP_TERMS else
                    f"{n_terms - TOP_TERMS} smaller terms", size=SMALL, color="ink2")
        elif r.kind == "intercept":
            cv.text(x_term, base, "intercept")
            cv.text(x_calc, base, "fitted bias", size=SMALL, color="ink2")
        else:  # the sum of the rows above
            cv.text(x_term, base, "Signed decision score", weight="semibold")
        if r.kind == "total":
            cv.bar(sx(0), sx(r.contribution), cy, bar_h, "total", square_at=sx(0))
            label = "above 0, so labelled positive" if r.contribution > 0 else "0 or below, so labelled negative"
            cv.text(sx(r.contribution) + 8, base, label, size=SMALL, color="ink2", knockout=True)
            cv.text(x_val, base, signed(r.contribution, 3), weight="semibold", ha="right")
        else:
            start, end = run[i], run[i] + r.contribution
            if r.contribution != 0:
                cv.bar(sx(start), sx(end), cy, bar_h, "pos" if r.contribution > 0 else "neg")
            if i + 1 < len(table) - 1:  # connector to the next step's start
                cv.line([sx(end)] * 2, [cy + bar_h / 2, cy + row_h - bar_h / 2], color="axis", z=2)
            cv.text(x_val, base, "0" if r.contribution == 0 else signed(r.contribution, 3), ha="right")
        y += row_h
    return cv


def draw_length_narrow(table, theme):
    """Figure 07 for a phone-width column: each length quarter's label above its two intervals."""
    m, xmax, group_h = 8, 0.25, 66
    right = NARROW - m
    x_rate = right - 72  # right edge of the rate column; errors / reviews end at `right`
    px0, px1 = m + 2, x_rate - 52

    def sx(v):
        return px0 + v / xmax * (px1 - px0)

    title = wrap("Class-conditional error rates across review-length quartiles", right - m, TITLE, "semibold")
    subtitle = wrap("Historical model, internal holdout (2,000 reviews). Error rate by true class per length quartile: "
                    "the negative-class rate is higher in all four; no clear trend with length.", right - m, SMALL)
    legend = [(cls, wrap(label, right - m - 15, SMALL)) for cls, label in (
        ("negative", "Negative reviews (true label), wrongly called positive"),
        ("positive", "Positive reviews (true label), wrongly called negative"))]
    footnote = wrap("Exploratory. Quarters split all 2,000 held-out reviews by word count. Each 95% Wilson interval "
                    "covers one cell, treats reviews as independent and is not adjusted for multiple comparisons.",
                    right - m, SMALL)
    groups = list(dict.fromkeys(table["length_group"]))
    y_sub = 22 + 21 * (len(title) - 1) + 20
    y_legend = y_sub + 16 * (len(subtitle) - 1) + 24
    head_y = y_legend + 16 * (sum(len(lines) for _, lines in legend) - 1) + 42  # two header lines end here
    top = head_y + 9
    bottom = top + group_h * len(groups) + 8
    cv = Canvas(bottom + 40 + 16 * (len(footnote) - 1) + 12, theme, width=NARROW, scale=NARROW_SCALE)

    y = 22
    for line in title:
        cv.text(m, y, line, size=TITLE, weight="semibold")
        y += 21
    y = y_sub
    for line in subtitle:
        cv.text(m, y, line, size=SMALL, color="ink2")
        y += 16
    y = y_legend
    for cls, lines in legend:
        cv.marker(m + 5, y - 4.5, cls)
        for line in lines:
            cv.text(m + 15, y, line, size=SMALL, color="ink2")
            y += 16
    for x, y_h, label, ha in ((m, head_y, "Error rate, 95% Wilson interval", "left"), (x_rate, head_y, "Rate", "right"),
                              (right, head_y - 15, "Errors /", "right"), (right, head_y, "reviews", "right")):
        cv.text(x, y_h, label, size=SMALL, color="ink2", weight="semibold", ha=ha)
    for v in np.arange(0, xmax + 1e-9, 0.05):
        cv.line([sx(v)] * 2, [bottom, bottom + 4], color="axis")
        cv.text(sx(v), bottom + 18, f"{v:.0%}", size=SMALL, color="ink2", ha="center")
    for i, line in enumerate(footnote):
        cv.text(m, bottom + 40 + 16 * i, line, size=SMALL, color="ink2")

    y = top
    cv.line([m, right], [y, y])
    for name in groups:
        if name == "all lengths":
            y += 8
            cv.line([m, right], [y, y], color="axis")
        rows = table[table["length_group"] == name]
        first = rows.iloc[0]
        span = f"{first.min_words:,}–{first.max_words:,} words"
        label, detail = (("All lengths", f"{span}, {int(rows.reviews.sum()):,} reviews") if name == "all lengths"
                         else (span, f"{name}, {int(rows.reviews.sum()):,} reviews"))
        weight = "semibold" if name == "all lengths" else "normal"
        cv.text(m, y + 19, label, weight=weight)
        cv.text(m + text_width(label, BODY, weight) + 8, y + 19, detail, size=SMALL, color="ink2")
        rows_mid = [y + 37 + 18 * j for j in range(len(rows))]
        for v in np.arange(0, xmax + 1e-9, 0.05):  # this group's gridlines, clear of its label line
            cv.line([sx(v)] * 2, [rows_mid[0] - 9, rows_mid[-1] + 9], color="axis" if v == 0 else "grid")
        for cy, r in zip(rows_mid, rows.itertuples()):
            cv.line([sx(r.ci95_low), sx(r.ci95_high)], [cy, cy], color=MARK[r.true_class][1], width=2, z=2, cap="round")
            cv.marker(sx(r.error_rate), cy, r.true_class)
            cv.text(x_rate, cy + 4.5, f"{r.error_rate:.1%}", size=SMALL, ha="right")
            cv.text(right, cy + 4.5, f"{r.errors:,} / {r.reviews:,}", size=SMALL, ha="right")
        y += group_h
        if name != "all lengths":
            cv.line([m, right], [y, y])
    return cv


def draw_contributions_narrow(table, text, n_terms, theme):
    """Figure 08 for a phone-width column: each term over its weight x TF-IDF, beside its running-total bar."""
    m, row_h, total_h, bar_h = 8, 38, 46, 12
    right = NARROW - m
    run = [0.0] + table["running_total"].tolist()[:-1]
    lo = min(0.0, min(run)) - 0.05
    hi = max(run) + 0.05

    def cell(r):
        """The left cell: what the row is, then how its contribution is made."""
        if r.kind == "term":  # four decimals, so the shown factors multiply back to the shown contribution
            return r.component, "semibold", f"{signed(r.weight, 4)} × {r.tfidf:.4f}"
        if r.kind == "other":
            return "other terms", "normal", (f"none (all {n_terms} shown)" if n_terms <= TOP_TERMS
                                             else f"{n_terms - TOP_TERMS} smaller terms")
        return "intercept", "normal", "fitted bias"

    rows = [r for r in table.itertuples() if r.kind != "total"]
    total = next(r for r in table.itertuples() if r.kind == "total")
    cells = [cell(r) for r in rows]
    x_bar0 = m + max(max(text_width(name, BODY, w), text_width(calc, SMALL)) for name, w, calc in cells) + 14
    x_bar1 = right - 50

    def sx(v):
        return x_bar0 + (v - lo) / (hi - lo) * (x_bar1 - x_bar0)

    title = wrap(f"Additive decomposition of the linear decision function: “{text}”", right - m, TITLE, "semibold")
    subtitle = wrap("The score is the intercept plus each vocabulary word's SVM weight × its TF-IDF value in the text. "
                    "Bars add the terms in order of size. These are contributions within this linear model, not "
                    "causal explanations of language.", right - m, SMALL)
    y_sub = 22 + 21 * (len(title) - 1) + 20
    head_y = y_sub + 16 * (len(subtitle) - 1) + 42  # two header lines end here
    top = head_y + 9
    total_top = top + row_h * len(rows) + 8
    bottom = total_top + total_h
    cv = Canvas(bottom + 30, theme, width=NARROW, scale=NARROW_SCALE)

    y = 22
    for line in title:
        cv.text(m, y, line, size=TITLE, weight="semibold")
        y += 21
    y = y_sub
    for line in subtitle:
        cv.text(m, y, line, size=SMALL, color="ink2")
        y += 16
    cv.text(m, head_y - 15, "Term", size=SMALL, color="ink2", weight="semibold")
    cv.text(m, head_y, "weight × TF-IDF", size=SMALL, color="ink2")
    cv.text(x_bar0, head_y, "Running total", size=SMALL, color="ink2", weight="semibold")
    cv.text(right, head_y, "Contribution", size=SMALL, color="ink2", weight="semibold", ha="right")
    for v in np.arange(0, hi, 0.25):
        cv.line([sx(v)] * 2, [top, bottom], color="axis" if v == 0 else "grid")
        cv.text(sx(v), bottom + 17, signed(v) if v else "0", size=SMALL, color="ink2", ha="center")

    y = top
    cv.line([m, right], [y, y])
    for i, (r, (name, weight, calc)) in enumerate(zip(rows, cells)):
        cy = y + 19
        cv.text(m, y + 16, name, weight=weight)
        cv.text(m, y + 31, calc, size=SMALL, color="ink2")
        start, end = run[i], run[i] + r.contribution
        if r.contribution != 0:
            cv.bar(sx(start), sx(end), cy, bar_h, "pos" if r.contribution > 0 else "neg")
        if i + 1 < len(rows):  # connector to the next step's start
            cv.line([sx(end)] * 2, [cy + bar_h / 2, cy + row_h - bar_h / 2], color="axis", z=2)
        cv.text(right, cy + 4.5, "0" if r.contribution == 0 else signed(r.contribution, 3), ha="right")
        y += row_h

    cv.line([m, right], [total_top, total_top], color="axis")
    cv.text(m, total_top + 17, "Signed decision score", weight="semibold", knockout=True)
    cv.text(right, total_top + 17, signed(total.contribution, 3), weight="semibold", ha="right")
    cy = total_top + 32
    cv.bar(sx(0), sx(total.contribution), cy, bar_h, "total", square_at=sx(0))
    verdict = "above 0, so labelled positive" if total.contribution > 0 else "0 or below, so labelled negative"
    if sx(total.contribution) + 8 + text_width(verdict, SMALL) > right:
        verdict = "above 0: positive" if total.contribution > 0 else "0 or below: negative"
    cv.text(sx(total.contribution) + 8, cy + 4.5, verdict, size=SMALL, color="ink2", knockout=True)
    return cv


# ----------------------------------------------------------------------------- outputs

def sha256(path):
    """SHA-256 of a file. Text files are hashed with CRLF read as LF, so a Windows checkout gives the same value."""
    data = path.read_bytes()
    if path.suffix in {".csv", ".json", ".py"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


# The code and tracked data the figures depend on. models/ is not in git, so its files are hashed instead.
SOURCES = ["scripts/make_readme_figures.py", "src", "train.py", "data/training_data.csv", "results/historical/metrics.json"]


def git_state():
    """The commit the figures were made from, and whether the code and data they use match it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        changed = git("status", "--porcelain", "--", *SOURCES)
        return {"commit": git("rev-parse", "HEAD"), "sources": SOURCES, "sources_match_commit": not changed}
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    metrics, pipeline, manifest, held = load_and_check()
    version = manifest["model_version"]
    probes = score_probes(pipeline)
    length = error_by_length(held)
    parts, n_terms, gap = contributions(pipeline, PROBES[EXPLAINED][1])

    for table, name in ((probes, "06_probe_outputs.csv"), (length, "07_error_by_length.csv"),
                        (parts, "08_score_contributions.csv")):
        table.to_csv(OUT / name, index=False, lineterminator="\n")  # full precision
    written = []
    for theme in THEMES:
        written.append(draw_probes(probes, version, theme).save(f"06_probe_outputs_{theme}.png"))
        written.append(draw_probes_narrow(probes, version, theme).save(f"06_probe_outputs_narrow_{theme}.png"))
        written.append(draw_length(length, theme).save(f"07_error_by_length_{theme}.png"))
        written.append(draw_length_narrow(length, theme).save(f"07_error_by_length_narrow_{theme}.png"))
        written.append(draw_contributions(parts, PROBES[EXPLAINED][1], n_terms, theme)
                       .save(f"08_score_contributions_{theme}.png"))
        written.append(draw_contributions_narrow(parts, PROBES[EXPLAINED][1], n_terms, theme)
                       .save(f"08_score_contributions_narrow_{theme}.png"))

    model_files = [f"models/artefacts/{version}/pipeline.joblib"]
    inputs = model_files + ["results/historical/metrics.json", "models/test_predictions.csv", "data/training_data.csv"]

    def narrow(stem):
        return (f"{stem}_narrow_{{light,dark}}.png: the same data, {NARROW} CSS px wide at {NARROW_SCALE}x, "
                "for README columns on viewports below 960 px")

    provenance = {
        "generated_by": "scripts/make_readme_figures.py",
        "script_sha256": sha256(Path(__file__).resolve()),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_state(),
        "model_version": version,
        "inputs_sha256": {name: sha256(ROOT / name) for name in inputs},
        "sha256_note": "text files (.csv, .json, .py) are hashed with CRLF read as LF; model_version is the first 12 "
                       "hex characters of the Pipeline artefact's SHA-256, as the API reports it",
        "legacy_model_version": "98bcdfb288fc",
        "legacy_equivalence": "results/historical/refit_equivalence.json: parameters identical to the original two-file artefact",
        "checks_passed": [
            "seeded split (seed, test_size from metrics.json) reproduces test_predictions.csv row for row",
            "saved model reproduces all 2,000 saved held-out predictions",
            "recomputed accuracy and confusion matrix equal results/historical/metrics.json",
            f"figure 08 components sum to the model's decision score (absolute difference {gap:.1e})",
        ],
        "figures": {
            "06_probe_outputs": {"data": "06_probe_outputs.csv", "derived_from": model_files + ["PROBES in the script"],
                                 "note": "hand-written probes; expected = the sentiment each was written to express",
                                 "narrow_layout": narrow("06_probe_outputs")},
            "07_error_by_length": {"data": "07_error_by_length.csv",
                                   "derived_from": ["models/test_predictions.csv (text, true label, prediction)"],
                                   "verified_against": [name for name in inputs if name != "models/test_predictions.csv"],
                                   "binning": "quartiles of whitespace word count over all 2,000 held-out reviews "
                                              "(pandas.qcut, both classes pooled)",
                                   "interval": "95% Wilson score interval per cell; treats reviews as independent",
                                   "narrow_layout": narrow("07_error_by_length")},
            "08_score_contributions": {"data": "08_score_contributions.csv", "derived_from": model_files,
                                       "text": PROBES[EXPLAINED][1],
                                       "method": "contribution = SVM weight x TF-IDF value; plus intercept; "
                                                 f"terms beyond the largest {TOP_TERMS} summed as other terms",
                                       "narrow_layout": narrow("08_score_contributions")},
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                        "font": FONT[0]},
    }
    (OUT / "figures_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    print("wrote visual_outputs/06_probe_outputs.csv, 07_error_by_length.csv, 08_score_contributions.csv, "
          "figures_provenance.json")


if __name__ == "__main__":
    main()
