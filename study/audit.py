"""Duplicate, near-duplicate and grouping audit of the official partitions (texts only; no model output).

    python -m study.audit

Reads data/derived/imdb_official.csv.gz (study/data.py) and writes
  results/data/duplicate_audit.json      exact and normalised duplicates within and across partitions,
                                         label conflicts, the near-duplicate screen and the id-order profile
  results/data/test_overlap.csv.gz       per test review: exact/normalised overlap with the training
                                         partition and the highest TF-IDF cosine to any training review

Normalisation: HTML entities decoded, tags removed, NFKD with combining marks dropped, lower-cased,
every run of non-alphanumeric characters collapsed to one space.
Near-duplicate screen: cosine similarity of sublinear-TF TF-IDF vectors over word unigrams and
bigrams (min_df=2, max_df=0.5, fitted on both partitions' texts, which uses no labels). It is a
screen with heuristic thresholds, not a paraphrase detector.
Id-order profile: mean cosine between reviews whose ids differ by a given lag within one
(partition, label) directory, against random pairs from the same directory. A profile far above the
random level at short lags indicates that ids are ordered by movie.
"""
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived" / "imdb_official.csv.gz"
RESULTS = ROOT / "results" / "data"
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
LAGS = (1, 2, 5, 10, 20, 30, 50, 100, 200)
SEED = 42
GZIP = {"method": "gzip", "mtime": 0}


def normalise(text: str) -> str:
    text = re.sub(r"<[^<>]{0,200}>", " ", html.unescape(text))
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def duplicate_summary(df: pd.DataFrame, key: str) -> dict:
    out = {}
    for partition in ("train", "test"):
        part = df[df["partition"] == partition]
        sizes = part.groupby(key).size()
        groups = sizes[sizes > 1]
        conflicts = part[part[key].isin(groups.index)].groupby(key)["label"].nunique()
        out[partition] = {"reviews": len(part), "distinct_texts": int(sizes.size), "duplicate_groups": int(groups.size),
                          "reviews_in_duplicate_groups": int(groups.sum()), "redundant_copies": int((groups - 1).sum()),
                          "groups_with_conflicting_labels": int((conflicts > 1).sum())}
    train, test = df[df["partition"] == "train"], df[df["partition"] == "test"]
    shared = set(train[key]) & set(test[key])
    in_train = test[key].isin(shared)
    train_labels = train.groupby(key)["label"].agg(lambda s: set(s))
    conflicting = sum(1 for k, label in zip(test.loc[in_train, key], test.loc[in_train, "label"])
                      if train_labels[k] != {label})
    out["across"] = {"distinct_texts_in_both": len(shared), "test_reviews_whose_text_occurs_in_train": int(in_train.sum()),
                     "of_which_label_differs_from_train_copy": conflicting}
    return out


def near_duplicates(df: pd.DataFrame, chunk: int = 500) -> pd.DataFrame:
    train = df[df["partition"] == "train"].reset_index(drop=True)
    test = df[df["partition"] == "test"].reset_index(drop=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
    vectorizer.fit(pd.concat([train["norm"], test["norm"]]))
    x_train, x_test = vectorizer.transform(train["norm"]), vectorizer.transform(test["norm"])
    best, nearest = np.zeros(len(test), dtype=np.float32), np.zeros(len(test), dtype=np.int64)
    x_train_t = x_train.T.tocsr()
    for start in range(0, len(test), chunk):
        sims = (x_test[start:start + chunk] @ x_train_t).toarray()
        nearest[start:start + chunk] = sims.argmax(axis=1)
        best[start:start + chunk] = sims.max(axis=1)
    return pd.DataFrame({"example_id": test["example_id"], "max_cosine_to_train": best.round(6),
                         "nearest_train_id": train["example_id"].to_numpy()[nearest]})


def id_order_profile(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    profile = {}
    for (partition, label), part in df.groupby(["partition", "label"]):
        part = part.sort_values("review_id")
        x = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True).fit_transform(part["norm"])
        row = {f"lag_{lag}": float(np.asarray(x[:-lag].multiply(x[lag:]).sum(axis=1)).mean()) for lag in LAGS}
        perm = rng.permutation(x.shape[0])
        row["random_pairs"] = float(np.asarray(x[perm[:-1]].multiply(x[perm[1:]]).sum(axis=1)).mean())
        profile[f"{partition}/{'pos' if label else 'neg'}"] = {k: round(v, 4) for k, v in row.items()}
    return profile


def main() -> None:
    df = pd.read_csv(DERIVED)
    df["exact"] = [digest(t) for t in df["text"]]
    df["norm"] = [normalise(t) for t in df["text"]]
    df["normalised"] = [digest(t) for t in df["norm"]]
    near = near_duplicates(df)
    train_exact = set(df.loc[df["partition"] == "train", "exact"])
    train_norm = set(df.loc[df["partition"] == "train", "normalised"])
    test = df[df["partition"] == "test"].reset_index(drop=True)
    overlap = pd.DataFrame({"example_id": test["example_id"], "exact_in_train": test["exact"].isin(train_exact),
                            "normalised_in_train": test["normalised"].isin(train_norm)}).merge(near, on="example_id")
    audit = {
        "exact_duplicates": duplicate_summary(df, "exact"),
        "normalised_duplicates": duplicate_summary(df, "normalised"),
        "normalisation": "HTML entities decoded, tags removed, NFKD without combining marks, lower-cased, "
                         "non-alphanumeric runs collapsed to one space",
        "near_duplicate_screen": {
            "method": "max cosine of each test review to any training review; sublinear TF-IDF, word 1-2 grams, "
                      "min_df=2, max_df=0.5, fitted on both partitions' normalised texts (no labels)",
            "test_reviews_at_or_above": {str(t): int((overlap["max_cosine_to_train"] >= t).sum()) for t in THRESHOLDS},
            "max_cosine_quantiles": {q: float(np.quantile(overlap["max_cosine_to_train"], float(q)))
                                     for q in ("0.5", "0.9", "0.99", "0.999")},
        },
        "id_order_profile": {
            "method": "mean cosine (sublinear TF-IDF unigrams, min_df=2, max_df=0.5, fitted per directory) between "
                      "reviews whose ids differ by the lag, against random pairs from the same directory",
            "by_directory": id_order_profile(df),
        },
        "limitations": [
            "No movie identifiers are available, so same-movie dependence within a partition can only be approximated "
            "by contiguous id blocks.",
            "The screen measures lexical overlap; paraphrased or translated duplicates fall outside it.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "duplicate_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    overlap.to_csv(RESULTS / "test_overlap.csv.gz", index=False, lineterminator="\n", compression=GZIP)
    print(json.dumps({k: audit[k] for k in ("exact_duplicates", "normalised_duplicates")}, indent=1))
    print(json.dumps(audit["near_duplicate_screen"], indent=1))
    print(json.dumps(audit["id_order_profile"]["by_directory"], indent=1))


if __name__ == "__main__":
    main()
