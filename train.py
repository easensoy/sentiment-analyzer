"""Train the TF-IDF + linear SVC sentiment model and save both fitted artefacts.

Run from anywhere (paths are relative to this file):

    python train.py
    python train.py --data data/training_data.csv --out models --test-size 0.2 --seed 42

The CSV needs two columns: text and label (0 = negative, 1 = positive).
The vectoriser is fitted on the training split only, so the held-out split never
leaks into the vocabulary or IDF weights. The preprocessor and model are saved
together from the same run, so src/api.py always loads a matching pair.

Outputs in --out:
    model.joblib            fitted SVC (SentimentModel.save)
    preprocessor.joblib     fitted TextPreprocessor (cleaning + TF-IDF vocabulary)
    metrics.json            held-out accuracy with a 95% interval, F1, per-class report, confusion matrix
    test_predictions.csv    held-out text, true label and prediction
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.model import LABELS, SentimentModel
from src.preprocessor import TextPreprocessor

ROOT = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description="Train and evaluate the sentiment model")
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

    preprocessor = TextPreprocessor()
    model = SentimentModel()
    start = time.perf_counter()
    model.train(preprocessor.fit_transform(X_train), y_train)
    train_seconds = time.perf_counter() - start
    y_pred = model.predict(preprocessor.transform(X_test))

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
        "vocabulary_size": len(preprocessor.vectorizer.vocabulary_),
        "support_vectors": int(model.model.n_support_.sum()),
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
    model.save(args.out / "model.joblib")
    joblib.dump(preprocessor, args.out / "preprocessor.joblib")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    pd.DataFrame({"text": X_test, "label": y_test, "predicted": y_pred}).to_csv(
        args.out / "test_predictions.csv", index=False, lineterminator="\n"
    )

    print(f"train {len(X_train):,} / test {len(X_test):,} rows, vocabulary {metrics['vocabulary_size']:,} terms")
    print(f"SVC fitted in {train_seconds:.1f}s with {metrics['support_vectors']:,} support vectors")
    print(f"held-out accuracy {metrics['accuracy']:.4f} (95% CI {low:.4f}-{high:.4f}), "
          f"macro F1 {metrics['macro_f1']:.4f}")
    print(classification_report(y_test, y_pred, labels=labels, target_names=names, digits=4))
    print(f"saved model.joblib, preprocessor.joblib, metrics.json, test_predictions.csv to {args.out}")


if __name__ == "__main__":
    main()
