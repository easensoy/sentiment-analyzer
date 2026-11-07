"""Render the evaluation charts into visual_outputs/.

Run from the repo root after `python train.py`:

    python scripts/make_visuals.py

Reads data/training_data.csv and models/ (model.joblib, preprocessor.joblib,
metrics.json) and writes:

    visual_outputs/01_class_balance.png
    visual_outputs/02_confusion_matrix.png
    visual_outputs/03_per_class_metrics.png
    visual_outputs/04_top_terms.png
    visual_outputs/top_terms.csv        the numbers behind chart 4

Colours, marks and ink follow the dataviz reference palette (light mode):
negative = red pole #e34948, positive = blue pole #2a78d6, confusion counts on the
one-hue blue ramp. Needs matplotlib (dev-only, not in requirements.txt).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import BoxStyle, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # joblib must import src.preprocessor to unpickle it

MODELS = ROOT / "models"
OUT = ROOT / "visual_outputs"
DPI = 200

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WHITE = "#e1e0d9", "#c3c2b7", "#ffffff"
NEG, POS = "#e34948", "#2a78d6"  # diverging poles: negative / positive sentiment
NEG_MARK, POS_MARK = "s", "o"  # a second cue besides colour: square = negative, circle = positive
BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec",
        350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab",
        600: "#184f95", 650: "#104281", 700: "#0d366b"}

plt.rcParams.update({
    "font.family": ["Segoe UI"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.75,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 9.5,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
})


def css_px(n: float) -> float:
    """CSS pixels -> device pixels at the export DPI."""
    return n * DPI / 96


def figure(width_in, height_in, title, subtitle):
    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    fig.text(0.35 / width_in, 1 - 0.30 / height_in, title, fontsize=14.5, fontweight="semibold", va="top")
    fig.text(0.35 / width_in, 1 - 0.64 / height_in, subtitle, fontsize=9.5, color=INK_2, va="top")
    return fig


def axes(fig, left_in, bottom_in, width_in, height_in):
    w, h = fig.get_size_inches()
    ax = fig.add_axes([left_in / w, bottom_in / h, width_in / w, height_in / h])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    return ax


def legend(fig, x_in, y_in, items):
    """A row of marker + label keys in text ink (identity never by colour alone)."""
    w, h = fig.get_size_inches()
    handles = [Line2D([0], [0], marker=marker, linestyle="none", markersize=7.5, markerfacecolor=color,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=label) for label, color, marker in items]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x_in / w, y_in / h), ncol=len(items),
               frameon=False, fontsize=9.5, labelcolor=INK_2, handletextpad=0.3, columnspacing=1.6,
               borderaxespad=0)


def units_per_px(ax):
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(0, 0), (1, 1)])
    return abs(x1 - x0), abs(y1 - y0)


def rounded(ax, xy, width, height, color, radius=4, zorder=2):
    """Box with circular corners of `radius` CSS px, whatever the axis aspect."""
    ux, uy = units_per_px(ax)
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle=BoxStyle.Round(pad=0, rounding_size=css_px(radius) * ux),
                                mutation_aspect=uy / ux, facecolor=color, linewidth=0, zorder=zorder))


def bar(ax, center, end, color, thickness=24, horizontal=True, base=0.0):
    """Bar grown from a square baseline to a 4px rounded data end; thickness in CSS px."""
    ux, uy = units_per_px(ax)
    half = css_px(thickness) / 2 * (uy if horizontal else ux)
    lo, hi = sorted((base, end))
    cut = css_px(4) * (ux if horizontal else uy)
    square_end = end - cut if end > base else end + cut
    s_lo, s_hi = sorted((base, square_end))
    if horizontal:
        rounded(ax, (lo, center - half), hi - lo, 2 * half, color)
        ax.add_patch(Rectangle((s_lo, center - half), s_hi - s_lo, 2 * half, facecolor=color, linewidth=0, zorder=2))
    else:
        rounded(ax, (center - half, lo), 2 * half, hi - lo, color)
        ax.add_patch(Rectangle((center - half, s_lo), 2 * half, s_hi - s_lo, facecolor=color, linewidth=0, zorder=2))


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def class_balance(df, metrics):
    total = df["label"].value_counts().sort_index()
    cc = metrics["class_counts"]
    groups = [
        ("Full sample", f"{len(df):,} reviews", (int(total[0]), int(total[1]))),
        ("Training split", f"{metrics['rows']['train']:,} reviews (80%)", (cc["train"]["negative"], cc["train"]["positive"])),
        ("Held-out split", f"{metrics['rows']['test']:,} reviews (20%)", (cc["test"]["negative"], cc["test"]["positive"])),
    ]
    fig = figure(6.8, 4.9, "Exactly balanced: half negative, half positive in every split",
                 "Seeded sample of the IMDB train split (Maas et al., 2011); stratified 80/20 split, seed 42")
    legend(fig, 0.30, 4.9 - 0.95, [("Negative (label 0)", NEG, NEG_MARK), ("Positive (label 1)", POS, POS_MARK)])
    ax = axes(fig, 0.85, 0.80, 5.65, 2.95)
    top = max(max(c) for _, _, c in groups)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, top * 1.14)
    ax.yaxis.grid(True, color=GRID, linewidth=0.75)
    ax.spines["left"].set_visible(False)
    ux, uy = units_per_px(ax)
    step = (css_px(24) + css_px(2)) * ux / 2  # 24px bars, 2px surface gap between them
    for i, (_, _, counts) in enumerate(groups):
        for value, color, offset in ((counts[0], NEG, -step), (counts[1], POS, step)):
            bar(ax, i + offset, value, color, horizontal=False)
        # One label per pair: two counts side by side would not fit over 24px bars.
        label = f"{counts[0]:,} each" if counts[0] == counts[1] else f"{counts[0]:,} / {counts[1]:,}"
        ax.text(i, max(counts) + css_px(5) * uy, label, ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(groups)), [f"{name}\n{sub}" for name, sub, _ in groups])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Reviews")
    save(fig, "01_class_balance.png")


def confusion(metrics):
    cm = np.array(metrics["confusion_matrix"]["rows_true_cols_predicted"])
    names = [n.capitalize() for n in metrics["confusion_matrix"]["labels"]]
    fn, fp = cm[1, 0], cm[0, 1]
    fig = figure(6.6, 6.0, f"{fp} negative reviews were called positive; {fn} positive ones negative",
                 f"Held-out split, {cm.sum():,} reviews ({cm[0].sum():,} per class); accuracy {metrics['accuracy']:.4f}, "
                 f"macro F1 {metrics['macro_f1']:.4f}")
    ax = axes(fig, 1.55, 0.55, 4.4, 3.95)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.spines[["left", "bottom"]].set_visible(False)
    ux, uy = units_per_px(ax)
    gx, gy = css_px(1) * ux, css_px(1) * uy  # half of the 2px surface gap
    for r in range(2):
        for c in range(2):
            share = cm[r, c] / cm[r].sum()
            step = 100 + 50 * int(round(share * 12))
            ink = WHITE if step >= 450 else INK
            rounded(ax, (c + gx, r + gy), 1 - 2 * gx, 1 - 2 * gy, BLUE[step])
            ax.text(c + 0.5, r + 0.44, f"{cm[r, c]:,}", ha="center", va="center", fontsize=24,
                    fontweight="semibold", color=ink, zorder=3)
            verdict = "correct" if r == c else f"called {names[c].lower()}"
            ax.text(c + 0.5, r + 0.66, f"{share:.1%} of true {names[r].lower()}, {verdict}", ha="center",
                    va="center", fontsize=8.5, color=ink, zorder=3)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks([0.5, 1.5], names)
    ax.set_yticks([0.5, 1.5], names)
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label", labelpad=8)
    save(fig, "02_confusion_matrix.png")


def per_class(metrics):
    pc = metrics["per_class"]
    rows = [("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1-score")]
    values = [pc[c][k] for _, k in rows for c in ("negative", "positive")]
    n_neg, n_pos = int(pc["negative"]["support"]), int(pc["positive"]["support"])
    fig = figure(7.8, 4.3, f"Leans positive: recall {pc['positive']['recall']:.3f} on positive reviews, "
                           f"{pc['negative']['recall']:.3f} on negative",
                 f"Precision, recall and F1 per class on the held-out split; macro F1 {metrics['macro_f1']:.4f}, "
                 f"accuracy {metrics['accuracy']:.4f}")
    legend(fig, 0.30, 4.3 - 0.95, [(f"Negative (n = {n_neg:,})", NEG, NEG_MARK),
                                   (f"Positive (n = {n_pos:,})", POS, POS_MARK)])
    ax = axes(fig, 1.15, 0.65, 6.25, 2.45)
    lo = np.floor((min(values) - 0.015) * 50) / 50
    hi = np.ceil((max(values) + 0.015) * 50) / 50
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xticks(np.arange(lo, hi + 1e-9, 0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.xaxis.grid(True, color=GRID, linewidth=0.75)
    ax.spines["left"].set_visible(False)
    ux, _ = units_per_px(ax)
    pad = css_px(9) * ux
    for i, (_, key) in enumerate(rows):
        neg, pos = pc["negative"][key], pc["positive"][key]
        ax.plot([neg, pos], [i, i], color=AXIS, linewidth=1.5, solid_capstyle="round", zorder=1)
        for value, color, marker in ((neg, NEG, NEG_MARK), (pos, POS, POS_MARK)):
            ax.scatter([value], [i], s=72, marker=marker, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        left, right = sorted((neg, pos))
        ax.text(left - pad, i, f"{left:.3f}", ha="right", va="center", fontsize=9, color=INK)
        ax.text(right + pad, i, f"{right:.3f}", ha="left", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.tick_params(axis="y", labelcolor=INK, labelsize=10)
    ax.set_xlabel("Score (0 to 1)")
    save(fig, "03_per_class_metrics.png")


def top_terms(n=20):
    svc = joblib.load(MODELS / "model.joblib")
    pre = joblib.load(MODELS / "preprocessor.joblib")
    terms = pre.vectorizer.get_feature_names_out()
    coef = svc.coef_.toarray().ravel() if hasattr(svc.coef_, "toarray") else np.ravel(svc.coef_)
    # Positive weight pushes toward classes_[1]; check it rather than assume it.
    assert list(svc.classes_) == [0, 1], svc.classes_
    probe = pre.transform(["a wonderful, moving film", "a dull, terrible film"])
    assert np.allclose(probe @ coef + svc.intercept_[0], svc.decision_function(probe))

    order = np.argsort(coef)
    neg_idx, pos_idx = order[:n], order[::-1][:n]
    pd.DataFrame({
        "rank": range(1, n + 1),
        "negative_term": terms[neg_idx], "negative_weight": coef[neg_idx].round(4),
        "positive_term": terms[pos_idx], "positive_weight": coef[pos_idx].round(4),
    }).to_csv(OUT / "top_terms.csv", index=False, lineterminator="\n")

    height = 8.2
    fig = figure(9.4, height, "Terms with the largest SVC weights, toward each class",
                 f"Top {n} of {len(terms):,} TF-IDF terms per side. A term's effect on one review is its weight "
                 "times its TF-IDF value in that review")
    legend(fig, 0.30, height - 0.95, [("Weight toward negative", NEG, NEG_MARK),
                                      ("Weight toward positive", POS, POS_MARK)])
    ax = axes(fig, 0.35, 0.65, 8.7, height - 2.0)
    m = max(-coef[neg_idx].min(), coef[pos_idx].max())
    room = m * 0.42  # space beyond the longest bar for its label
    ax.set_xlim(-m - room, m + room)
    ax.set_ylim(n - 0.4, -0.6)
    ax.spines[["left", "bottom"]].set_visible(False)
    ticks = np.arange(-np.floor(m), np.floor(m) + 1, 1.0)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    ax.xaxis.grid(True, color=GRID, linewidth=0.75)
    ax.axvline(0, color=AXIS, linewidth=0.75, zorder=1)
    ax.set_yticks([])
    ux, _ = units_per_px(ax)
    pad = css_px(6) * ux
    gap = css_px(1) * ux  # each bar starts 1px off zero: a 2px surface gap where they meet
    for i, (k_neg, k_pos) in enumerate(zip(neg_idx, pos_idx)):
        bar(ax, i, coef[k_neg], NEG, thickness=18, base=-gap)
        bar(ax, i, coef[k_pos], POS, thickness=18, base=gap)
        ax.text(coef[k_neg] - pad, i, terms[k_neg], ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(coef[k_pos] + pad, i, terms[k_pos], ha="left", va="center", fontsize=9.5, color=INK)
    ax.set_xlabel("SVC weight (linear kernel); bars grow from 0, rank 1 at the top")
    save(fig, "04_top_terms.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    metrics = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / metrics["data"])
    class_balance(df, metrics)
    confusion(metrics)
    per_class(metrics)
    top_terms()


if __name__ == "__main__":
    main()
