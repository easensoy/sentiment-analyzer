"""Official IMDB partitions (Maas et al., 2011): verified sources, split membership, development folds.

    python -m study.data --archive data/raw/fastai-imdb/imdb.tgz --hf-dir data/raw/hf-imdb-e6281661

Sources, each pinned by size and SHA-256 (download commands in the README):
  * fast.ai's mirror of the aclImdb v1 archive. It carries each review's id and star rating in its
    file name; it lacks the urls_*.txt files that map reviews to movies. The canonical Stanford host
    (ai.stanford.edu) did not accept connections on 2026-11-08.
  * The Hugging Face stanfordnlp/imdb parquet files at revision e6281661, used for the historical sample.
The script stops unless both sources hold identical (text, label) multisets in each partition.

Outputs
  data/derived/imdb_official.csv.gz     every labelled review with its text (gitignored, never committed)
  results/data/split_membership.csv.gz  no text: ids, labels, ratings, lengths, 64-bit text hashes, bootstrap
                                        blocks, development folds and the historical-sample mapping
  results/data/source_manifest.json     sources, checksums, counts and verification results

Grouping. The archive README states that the two partitions contain disjoint sets of movies and at
most 30 reviews per movie, but no reachable source gives movie identifiers. Review ids are ordered by
movie within each (partition, label) directory (see study/audit.py), so contiguous id blocks serve as
an approximate grouping: 30-id blocks for the bootstrap, 100-id blocks for development folds.
"""
import argparse
import collections
import hashlib
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived" / "imdb_official.csv.gz"
RESULTS = ROOT / "results" / "data"
HF_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
HF_BASE = f"https://huggingface.co/datasets/stanfordnlp/imdb/resolve/{HF_REVISION}/plain_text"
SOURCES = {
    "archive": {"file": "imdb.tgz", "url": "https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz", "bytes": 144_440_600,
                "sha256": "d501018afa17aee9fa1ebe8ac29859a5609980e13dc6e611aa21567cc357351f",
                "description": "fast.ai mirror of aclImdb v1 (HTTP Last-Modified 2019-01-03)"},
    "hf_train": {"file": "train-00000-of-00001.parquet", "url": f"{HF_BASE}/train-00000-of-00001.parquet",
                 "bytes": 20_979_968, "sha256": "db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c",
                 "description": f"Hugging Face stanfordnlp/imdb, revision {HF_REVISION}"},
    "hf_test": {"file": "test-00000-of-00001.parquet", "url": f"{HF_BASE}/test-00000-of-00001.parquet",
                "bytes": 20_470_363, "sha256": "b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da",
                "description": f"Hugging Face stanfordnlp/imdb, revision {HF_REVISION}"},
}
BLOCK = 30  # bootstrap block, in consecutive review ids (the dataset allows at most 30 reviews per movie)
FOLD_BLOCK = 100  # development-fold grouping unit, in consecutive review ids
N_FOLDS = 5
SEED = 42
BR = re.compile(r"<br\s*/?>")  # the historical builder's pattern
GZIP = {"method": "gzip", "mtime": 0}  # byte-reproducible output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, spec: dict) -> dict:
    size, digest = path.stat().st_size, sha256_file(path)
    if size != spec["bytes"] or digest != spec["sha256"]:
        raise SystemExit(f"{path}: {size} bytes, SHA-256 {digest}; expected {spec['bytes']} bytes, {spec['sha256']}")
    return {**spec, "verified": True}


def read_archive(path: Path) -> tuple[pd.DataFrame, str]:
    """Labelled reviews from the archive: partition, label, review_id, rating, text; plus its README."""
    rows, readme = [], ""
    pattern = re.compile(r"imdb/(train|test)/(pos|neg)/(\d+)_(\d+)\.txt")
    with tarfile.open(path, "r:gz") as tar:  # read in memory; nothing is extracted to disk
        for member in tar:
            if not member.isfile():
                continue
            if member.name == "imdb/README":
                readme = tar.extractfile(member).read().decode("utf-8")
            elif match := pattern.fullmatch(member.name):
                text = tar.extractfile(member).read().decode("utf-8")
                rows.append((match[1], int(match[2] == "pos"), int(match[3]), int(match[4]), text))
    df = pd.DataFrame(rows, columns=["partition", "label", "review_id", "rating", "text"])
    df["example_id"] = df["partition"] + "/" + df["label"].map({0: "neg", 1: "pos"}) + "/" + df["review_id"].astype(str)
    return df.sort_values(["partition", "label", "review_id"]).reset_index(drop=True), readme


def verify_against_hf(df: pd.DataFrame, hf_dir: Path) -> dict:
    """Both sources must hold the same (text, label) multiset in each partition."""
    result = {}
    for partition, key in (("train", "hf_train"), ("test", "hf_test")):
        hf = pd.read_parquet(hf_dir / SOURCES[key]["file"], columns=["text", "label"])
        ours = df[df["partition"] == partition]
        same = collections.Counter(zip(ours["text"], ours["label"])) == collections.Counter(zip(hf["text"], hf["label"]))
        if not same:
            raise SystemExit(f"{partition}: the archive and the Hugging Face parquet differ")
        result[partition] = {"rows_archive": len(ours), "rows_hf": len(hf), "identical_text_label_multisets": True}
    return result


def development_folds(df: pd.DataFrame) -> pd.Series:
    """Training-partition fold per review: 100-id blocks, shuffled per label and dealt to folds in turn."""
    folds = pd.Series(-1, index=df.index, dtype=int)
    rng = np.random.default_rng(SEED)
    train = df["partition"] == "train"
    for label in (0, 1):
        rows = df[train & (df["label"] == label)]
        blocks = np.sort((rows["review_id"] // FOLD_BLOCK).unique())
        order = rng.permutation(blocks)
        fold_of = {block: i % N_FOLDS for i, block in enumerate(order)}
        folds.loc[rows.index] = (rows["review_id"] // FOLD_BLOCK).map(fold_of)
    return folds


def map_historical(df: pd.DataFrame, historical_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Locate the 10,000 historical sample rows (and their 8,000/2,000 split) among the official reviews."""
    hist = pd.read_csv(historical_csv)
    _, x_holdout = train_test_split(hist["text"], test_size=0.2, stratify=hist["label"], random_state=SEED)
    role = pd.Series("train", index=hist.index)
    role.loc[x_holdout.index] = "holdout"
    train = df[df["partition"] == "train"]
    by_key: dict[str, list[int]] = collections.defaultdict(list)
    for idx, text in zip(train.index, train["text"]):
        by_key[BR.sub(" ", text).strip()].append(idx)
    out = pd.DataFrame({"historical_row": pd.array([pd.NA] * len(df), dtype="Int64"), "historical_role": ""},
                       index=df.index)
    unmatched, ambiguous, label_mismatch = 0, 0, 0
    for row, text, label in zip(hist.index, hist["text"], hist["label"]):
        matches = by_key.get(text, [])
        if not matches:
            unmatched += 1
            continue
        ambiguous += len(matches) > 1
        target = min(matches, key=lambda i: df.at[i, "review_id"])  # deterministic choice among duplicates
        label_mismatch += int(df.at[target, "label"] != label)
        out.at[target, "historical_row"] = int(row)
        out.at[target, "historical_role"] = role[row]
    summary = {"sample_rows": len(hist), "matched": len(hist) - unmatched, "unmatched": unmatched,
               "matched_text_with_several_official_ids": ambiguous, "label_mismatches": label_mismatch,
               "historical_train_rows": int((out["historical_role"] == "train").sum()),
               "historical_holdout_rows": int((out["historical_role"] == "holdout").sum())}
    if unmatched or label_mismatch:
        raise SystemExit(f"historical sample does not map cleanly onto the official training partition: {summary}")
    return out, summary


def build(archive: Path, hf_dir: Path, historical_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sources = {"archive": verify_source(archive, SOURCES["archive"]),
               "hf_train": verify_source(hf_dir / SOURCES["hf_train"]["file"], SOURCES["hf_train"]),
               "hf_test": verify_source(hf_dir / SOURCES["hf_test"]["file"], SOURCES["hf_test"])}
    df, readme = read_archive(archive)
    agreement = verify_against_hf(df, hf_dir)
    df["text_sha256_64"] = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in df["text"]]
    df["n_words"] = [len(BR.sub(" ", t).split()) for t in df["text"]]
    df["id_block"] = df["review_id"] // BLOCK
    df["dev_fold"] = development_folds(df)
    hist, hist_summary = map_historical(df, historical_csv)
    df = pd.concat([df, hist], axis=1)
    membership = df.drop(columns=["text"])
    counts = df.groupby(["partition", "label"]).size()
    manifest = {
        "dataset": "Large Movie Review Dataset v1.0 (Maas et al., ACL 2011)",
        "sources": sources,
        "canonical_host_status": "ai.stanford.edu did not accept TCP connections on ports 80 or 443 on 2026-11-08 (curl: connection attempts timed out after about 21 s)",
        "cross_source_verification": agreement,
        "counts": {f"{p}/{'pos' if l else 'neg'}": int(n) for (p, l), n in counts.items()},
        "ratings": {p: sorted(int(r) for r in df.loc[df["partition"] == p, "rating"].unique()) for p in ("train", "test")},
        "readme_statements": [line.strip() for line in readme.splitlines()
                              if re.search(r"disjoint|no more than 30|urls_", line)],
        "movie_identifiers": "absent: the mirror omits urls_[pos,neg,unsup].txt; no reachable source provides them",
        "grouping_proxy": {"bootstrap_block_ids": BLOCK, "fold_block_ids": FOLD_BLOCK, "n_folds": N_FOLDS, "seed": SEED,
                           "rationale": "review ids are ordered by movie within each directory; see results/data/duplicate_audit.json"},
        "historical_sample": hist_summary,
        "text_hash": "first 16 hex characters of the SHA-256 of the UTF-8 review text as distributed",
        "length_definition": "whitespace-separated tokens after replacing <br /> tags with a space",
    }
    return df, membership, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "raw" / "fastai-imdb" / "imdb.tgz")
    parser.add_argument("--hf-dir", type=Path, default=ROOT / "data" / "raw" / "hf-imdb-e6281661")
    parser.add_argument("--historical", type=Path, default=ROOT / "data" / "training_data.csv")
    args = parser.parse_args()
    df, membership, manifest = build(args.archive, args.hf_dir, args.historical)
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df[["example_id", "partition", "label", "review_id", "rating", "text"]].to_csv(
        DERIVED, index=False, lineterminator="\n", compression=GZIP)
    membership.to_csv(RESULTS / "split_membership.csv.gz", index=False, lineterminator="\n", compression=GZIP)
    (RESULTS / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "historical_sample": manifest["historical_sample"]}, indent=2))
    print(f"wrote {DERIVED.relative_to(ROOT)} (not committed), results/data/split_membership.csv.gz, source_manifest.json")


if __name__ == "__main__":
    main()
