"""Experimental factors, development grids and the frozen candidate set.

Every candidate is one Pipeline configuration (src/pipeline.py) plus a training set. Candidates differ
from a reference in one factor at a time, so each paired comparison isolates one change:
estimator, feature representation, cleaning version or training-set size. The one combined
comparison (final branch candidate against the historical model) is labelled as combined.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "results" / "dev" / "selection.json"

LEGACY_UNI = {"cleaning": "legacy-v1", "ngram_range": [1, 1], "min_df": 1}
LEGACY_BI = {"cleaning": "legacy-v1", "ngram_range": [1, 2], "min_df": 1}
V2_BI = {"cleaning": "v2", "ngram_range": [1, 2], "min_df": 1}
FEATURE_SETS = {"legacy-v1/unigram": LEGACY_UNI, "legacy-v1/uni+bigram": LEGACY_BI, "v2/uni+bigram": V2_BI}

# log2 grids; the selection rule is the highest mean validation accuracy, ties to the smaller C
GRIDS = {"logreg": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
         "linear_svc": [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]}
SVC_C = 1.0  # the historical setting, not tuned: one libsvm fit on 20,000 reviews takes several minutes
TRAINING_SETS = {"full": "all 25,000 reviews of the official training partition",
                 "historical-8k": "the 8,000 training reviews of the historical model (a subset of the same partition)"}


def selection() -> dict:
    if not SELECTION.exists():
        raise SystemExit("results/dev/selection.json is missing; run: python -m study.develop")
    return json.loads(SELECTION.read_text(encoding="utf-8"))


def candidates(sel: dict | None = None) -> dict:
    """Frozen candidate set: id -> {config, training_set, role}. Tuned C values come from development."""
    sel = sel or selection()
    branch, c = sel["branch_estimator"], sel["selected_C"]
    return {
        "majority": {"config": None, "training_set": "full",
                     "role": "constant prediction of the training partition's majority class (ties: negative)"},
        "svc_uni_8k_historical": {"config": {**LEGACY_UNI, "estimator": "svc", "C": SVC_C}, "training_set": "historical-8k",
                                  "role": "the historical model (artefact re-fitted by train.py, parameter-identical)"},
        "svc_uni_full": {"config": {**LEGACY_UNI, "estimator": "svc", "C": SVC_C}, "training_set": "full",
                         "role": "historical configuration trained on the full partition"},
        "logreg_uni_full": {"config": {**LEGACY_UNI, "estimator": "logreg", "C": c["logreg"]["legacy-v1/unigram"]},
                            "training_set": "full", "role": "estimator comparison"},
        "linsvc_uni_full": {"config": {**LEGACY_UNI, "estimator": "linear_svc", "C": c["linear_svc"]["legacy-v1/unigram"]},
                            "training_set": "full", "role": "estimator comparison"},
        f"{short(branch)}_uni_8k": {"config": {**LEGACY_UNI, "estimator": branch, "C": c[branch]["legacy-v1/unigram"]},
                                    "training_set": "historical-8k", "role": "training-set size comparison"},
        f"{short(branch)}_bi_full": {"config": {**LEGACY_BI, "estimator": branch, "C": c[branch]["legacy-v1/uni+bigram"]},
                                     "training_set": "full", "role": "feature comparison"},
        f"{short(branch)}_bi_v2_full": {"config": {**V2_BI, "estimator": branch, "C": c[branch]["v2/uni+bigram"]},
                                        "training_set": "full", "role": "cleaning comparison"},
    }


def short(estimator: str) -> str:
    return {"logreg": "logreg", "linear_svc": "linsvc", "svc": "svc"}[estimator]


def comparisons(sel: dict | None = None) -> list[dict]:
    """Pre-specified paired comparisons: (candidate A minus candidate B) and the factor that differs."""
    sel = sel or selection()
    b = short(sel["branch_estimator"])
    return [
        {"id": "size_svc", "a": "svc_uni_full", "b": "svc_uni_8k_historical", "factor": "training-set size (SVC, C=1)"},
        {"id": f"size_{b}", "a": f"{b}_uni_full", "b": f"{b}_uni_8k", "factor": f"training-set size ({b})"},
        {"id": "estimator_linsvc_vs_logreg", "a": "linsvc_uni_full", "b": "logreg_uni_full",
         "factor": "estimator (both C-tuned on development folds)"},
        {"id": "estimator_svc_vs_linsvc", "a": "svc_uni_full", "b": "linsvc_uni_full",
         "factor": "estimator (SVC at the untuned historical C=1 vs C-tuned LinearSVC)"},
        {"id": "features_bigrams", "a": f"{b}_bi_full", "b": f"{b}_uni_full", "factor": "word 1-2 grams vs unigrams"},
        {"id": "cleaning_v2", "a": f"{b}_bi_v2_full", "b": f"{b}_bi_full", "factor": "cleaning v2 vs legacy-v1"},
        {"id": "combined_vs_historical", "a": f"{b}_bi_v2_full", "b": "svc_uni_8k_historical",
         "factor": "combined: estimator, C, features, cleaning and training-set size together"},
    ]
