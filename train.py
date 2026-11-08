"""Train the historical model (internal-holdout protocol) and publish it as a versioned Pipeline.

Run from anywhere (paths are relative to this file):

    python train.py
    python train.py --data data/training_data.csv --out models --test-size 0.2 --seed 42

Protocol (unchanged since 2026-11-06): data/training_data.csv holds a seeded 10,000-review sample of
the official IMDB training partition; a stratified 80/20 split with the given seed gives 8,000
training and 2,000 held-out reviews. The held-out reviews are not the official test partition, and
reviews of one movie can fall on both sides. The study in study/ evaluates on the official test.

The Pipeline is cleaning version legacy-v1 -> TfidfVectorizer() -> SVC(kernel="linear", C=1), fitted
on the training split only, so held-out reviews never reach the vocabulary or IDF weights. The same
Pipeline object serves requests, so training and serving share one cleaning path.

Outputs in --out:
    artefacts/<model_version>/{pipeline.joblib,manifest.json}  the fitted Pipeline and its manifest
    serving.json                                             atomic pointer to that artefact (src/api.py)
    metrics.json            held-out accuracy with a 95% interval, F1, per-class report, confusion matrix
    test_predictions.csv    held-out row index, text, true label, prediction and signed decision score
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.pipeline import LABELS, build_pipeline, labels_from_scores, n_features, publish

ROOT = Path(__file__).resolve().parent
CANDIDATE_ID = "historical-svc-c1-uni-legacy-v1-8k"
CONFIG = {"cleaning": "legacy-v1", "ngram_range": [1, 1], "min_df": 1, "estimator": "svc", "C": 1.0}


def wilson_interval(correct: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for accuracy. It treats the held-out reviews as independent."""
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z / (1 + z * z / total) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return centre - half, centre + half


def display_path(path: Path) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:  # a different drive on Windows
        return str(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate and publish the historical model")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "training_data.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    if df.empty or not {"text", "label"} <= set(df.columns):
        raise SystemExit(f"{args.data} needs rows with 'text' and 'label' columns")
    if not set(df["label"].unique()) <= set(LABELS):
        raise SystemExit(f"labels must be 0 (negative) or 1 (positive), got {sorted(df['label'].unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, stratify=df["label"], random_state=args.seed
    )

    pipeline = build_pipeline(CONFIG)
    start = time.perf_counter()
    pipeline.fit(X_train.tolist(), y_train.to_numpy())
    train_seconds = time.perf_counter() - start
    scores = pipeline[-1].decision_function(pipeline[:-1].transform(X_test.tolist()))
    y_pred = labels_from_scores(scores)

    labels = sorted(LABELS)
    names = [LABELS[label] for label in labels]
    correct = int((y_pred == y_test.to_numpy()).sum())
    low, high = wilson_interval(correct, len(y_test))
    metrics = {
        "data": display_path(args.data),
        "seed": args.seed,
        "test_size": args.test_size,
        "rows": {"train": len(X_train), "test": len(X_test)},
        "class_counts": {
            split: {LABELS[int(k)]: int(v) for k, v in y.value_counts().sort_index().items()}
            for split, y in (("train", y_train), ("test", y_test))
        },
        "vocabulary_size": n_features(pipeline),
        "support_vectors": int(pipeline[-1].n_support_.sum()),
        "train_seconds": round(train_seconds, 1),
        "accuracy": accuracy_score(y_test, y_pred),
        "accuracy_ci95": [round(low, 4), round(high, 4)],
        "accuracy_ci_method": "Wilson score interval; treats reviews as independent, so it ignores "
                              "reviews of the same movie landing on both sides of the split",
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "per_class": classification_report(y_test, y_pred, labels=labels, target_names=names, output_dict=True),
        "confusion_matrix": {"labels": names, "rows_true_cols_predicted": confusion_matrix(y_test, y_pred, labels=labels).tolist()},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    training = {
        "protocol": "historical internal holdout (stratified 80/20 split of a 10,000-review training-partition sample)",
        "data": display_path(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "rows": len(X_train), "seed": args.seed, "test_size": args.test_size,
        "train_seconds": round(train_seconds, 1),
    }
    manifest = publish(pipeline, args.out, candidate_id=CANDIDATE_ID, config=CONFIG, training=training)
    metrics["model_version"] = manifest["model_version"]
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    pd.DataFrame({"row": X_test.index, "text": X_test, "label": y_test, "predicted": y_pred, "score": scores}).to_csv(
        args.out / "test_predictions.csv", index=False, lineterminator="\n"
    )

    print(f"train {len(X_train):,} / test {len(X_test):,} rows, vocabulary {metrics['vocabulary_size']:,} terms")
    print(f"Pipeline fitted in {train_seconds:.1f}s with {metrics['support_vectors']:,} support vectors")
    print(f"held-out accuracy {metrics['accuracy']:.4f} (95% CI {low:.4f}-{high:.4f}), "
          f"macro F1 {metrics['macro_f1']:.4f}")
    print(classification_report(y_test, y_pred, labels=labels, target_names=names, digits=4))
    print(f"published model {manifest['model_version']} to {args.out}; wrote metrics.json, test_predictions.csv")


if __name__ == "__main__":
    main()
