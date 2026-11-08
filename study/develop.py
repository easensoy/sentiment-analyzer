"""Development-only model selection on the official training partition (grouped 5-fold cross-validation).

    python -m study.develop --jobs 4             # D1-D5 below (minutes)
    python -m study.develop --svc-reference      # SVC at C=1 on the same folds, for reference (slow)

Folds come from results/data/split_membership.csv.gz (dev_fold: 100-id blocks, dealt per label, so each
fold holds 2,500 reviews of each class and most reviews of one movie share a fold). The vectoriser is
fitted inside each training fold only. No test-partition review is read.

Selection rule, fixed before any test scoring: the C with the highest mean validation accuracy over
the five folds; ties go to the smaller C.
  D1  C for logistic regression and LinearSVC on legacy-v1 unigrams.
  D2  branch estimator: the D1 winner of the two (higher mean accuracy at its selected C).
  D3  C for both estimators on legacy-v1 word 1-2 grams.
  D4  C for both estimators on v2 word 1-2 grams.
  D5  v2 ablation for the branch estimator at its D4 C: each v2 component removed in turn
      (interpretation only; it selects nothing).
Writes results/dev/cv_folds.csv (one row per fit) and results/dev/selection.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

from src.cleaning import CLEANERS
from src.pipeline import make_estimator
from study.configs import FEATURE_SETS, GRIDS, LEGACY_UNI, SVC_C, V2_BI

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived" / "imdb_official.csv.gz"
MEMBERSHIP = ROOT / "results" / "data" / "split_membership.csv.gz"
OUT = ROOT / "results" / "dev"
N_FOLDS = 5


def load_train() -> pd.DataFrame:
    membership = pd.read_csv(MEMBERSHIP, usecols=["example_id", "partition", "dev_fold"])
    texts = pd.read_csv(DERIVED, usecols=["example_id", "partition", "label", "text"])
    texts = texts[texts["partition"] == "train"].merge(membership[membership["partition"] == "train"][["example_id", "dev_fold"]],
                                                     on="example_id", validate="one_to_one")
    if sorted(texts["dev_fold"].unique()) != list(range(N_FOLDS)) or len(texts) != 25_000:
        raise SystemExit("unexpected development folds")
    return texts


def fold_task(train: pd.DataFrame, cleaned: list[str], features: dict, estimator: str, fold: int, cs: list[float],
              feature_name: str) -> list[dict]:
    tr, va = train["dev_fold"].to_numpy() != fold, train["dev_fold"].to_numpy() == fold
    vectorizer = TfidfVectorizer(ngram_range=tuple(features["ngram_range"]), min_df=features["min_df"])
    texts = np.asarray(cleaned, dtype=object)
    x_tr, x_va = vectorizer.fit_transform(texts[tr]), vectorizer.transform(texts[va])
    y_tr, y_va = train["label"].to_numpy()[tr], train["label"].to_numpy()[va]
    rows = []
    for c in cs:
        model = make_estimator(estimator, c)
        start = time.perf_counter()
        model.fit(x_tr, y_tr)
        seconds = time.perf_counter() - start
        pred = (model.decision_function(x_va) > 0).astype(int)
        rows.append({"features": feature_name, "cleaning": features["cleaning"], "estimator": estimator, "C": c,
                     "fold": fold, "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_features": x_tr.shape[1],
                     "accuracy": float((pred == y_va).mean()),
                     "error_negative": float((pred[y_va == 0] != 0).mean()),
                     "error_positive": float((pred[y_va == 1] != 1).mean()), "fit_seconds": round(seconds, 3)})
    return rows


def select(rows: pd.DataFrame) -> tuple[float, float, float]:
    """(C, mean accuracy, SD across folds) with the highest mean accuracy; ties to the smaller C."""
    means = rows.groupby("C")["accuracy"].agg(["mean", "std"]).reset_index().sort_values(["mean", "C"], ascending=[False, True])
    best = means.iloc[0]
    return float(best["C"]), float(best["mean"]), float(best["std"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--svc-reference", action="store_true", help="only cross-validate SVC at C=1 (slow)")
    args = parser.parse_args()
    train = load_train()
    OUT.mkdir(parents=True, exist_ok=True)
    cleaned = {name: [CLEANERS[name](t) for t in train["text"]] for name in sorted(CLEANERS)}

    if args.svc_reference:
        rows = Parallel(n_jobs=min(args.jobs, N_FOLDS))(
            delayed(fold_task)(train, cleaned["legacy-v1"], LEGACY_UNI, "svc", fold, [SVC_C], "legacy-v1/unigram")
            for fold in range(N_FOLDS))
        pd.DataFrame([r for rs in rows for r in rs]).to_csv(OUT / "cv_svc_reference.csv", index=False, lineterminator="\n")
        print(pd.DataFrame([r for rs in rows for r in rs])[["fold", "accuracy", "fit_seconds"]])
        return

    tasks = [(feature_name, features, estimator, fold)
             for feature_name, features in FEATURE_SETS.items() for estimator in GRIDS for fold in range(N_FOLDS)]
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fold_task)(train, cleaned[features["cleaning"]], features, estimator, fold, GRIDS[estimator], feature_name)
        for feature_name, features, estimator, fold in tasks)
    cv = pd.DataFrame([r for rs in results for r in rs])

    selected, summary = {}, {}
    for (estimator, feature_name), rows in cv.groupby(["estimator", "features"]):
        c, mean, sd = select(rows)
        selected.setdefault(estimator, {})[feature_name] = c
        summary.setdefault(estimator, {})[feature_name] = {"C": c, "mean_accuracy": round(mean, 5), "sd_accuracy": round(sd, 5)}
    uni = {e: summary[e]["legacy-v1/unigram"]["mean_accuracy"] for e in GRIDS}
    branch = max(sorted(uni), key=lambda e: uni[e])  # ties: alphabetical order, so the rule stays deterministic

    # D5: v2 ablation for the branch estimator at its D4 C (interpretation only)
    ablation_versions = [v for v in sorted(CLEANERS) if v.startswith("v2-no-")]
    ablation_rows = Parallel(n_jobs=args.jobs)(
        delayed(fold_task)(train, [CLEANERS[v](t) for t in train["text"]], {**V2_BI, "cleaning": v}, branch, fold,
                           [selected[branch]["v2/uni+bigram"]], f"{v}/uni+bigram")
        for v in ablation_versions for fold in range(N_FOLDS))
    cv = pd.concat([cv, pd.DataFrame([r for rs in ablation_rows for r in rs])], ignore_index=True)
    cv.to_csv(OUT / "cv_folds.csv", index=False, lineterminator="\n")
    ablation = (cv[cv["cleaning"].isin(ablation_versions + ["v2"]) & (cv["estimator"] == branch)
                   & (cv["C"] == selected[branch]["v2/uni+bigram"])]
                .groupby("cleaning")["accuracy"].agg(["mean", "std"]).round(5))

    sel = {
        "rule": "highest mean validation accuracy over 5 grouped folds; ties to the smaller C",
        "folds": "100-id blocks per label dealt to 5 folds (seed 42); 2,500 reviews per class per fold",
        "grids": GRIDS,
        "selected_C": selected,
        "summary": summary,
        "branch_estimator_rule": "higher D1 mean accuracy among logreg and linear_svc on legacy-v1 unigrams",
        "branch_estimator": branch,
        "v2_ablation_at_selected_C": {k: {"mean_accuracy": float(r["mean"]), "sd_accuracy": float(r["std"])}
                                      for k, r in ablation.iterrows()},
    }
    (OUT / "selection.json").write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
