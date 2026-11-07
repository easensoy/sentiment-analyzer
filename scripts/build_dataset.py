"""Build data/training_data.csv: a seeded, class-balanced sample of the IMDB train split.

Source: Large Movie Review Dataset (Maas et al., ACL 2011), official 25,000-review
train split, read from the Hugging Face mirror (the Stanford host was unreachable):

    dataset   https://huggingface.co/datasets/stanfordnlp/imdb
    revision  e6281661ce1c48d982bc483cf8a173c1bbeb5d31
    file      plain_text/train-00000-of-00001.parquet  (20,979,968 bytes)

Download the file, then run from the repo root:

    curl -L -o imdb_train.parquet https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text/train-00000-of-00001.parquet
    python scripts/build_dataset.py --parquet imdb_train.parquet

Output columns: text, label (0 = negative, 1 = positive). HTML <br /> tags are
replaced by a space; all other cleaning stays in src/preprocessor.py.
Needs pyarrow to read parquet (a dev-only tool, not in requirements.txt).
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA256 = "db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample IMDB reviews into data/training_data.csv")
    parser.add_argument("--parquet", type=Path, required=True, help="IMDB train parquet file")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "training_data.csv")
    parser.add_argument("--per-class", type=int, default=5000, help="reviews per label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-checksum-mismatch", action="store_true",
                        help="build from a file other than the pinned revision (the sample will differ)")
    args = parser.parse_args()

    digest = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    if digest != PINNED_SHA256:
        problem = f"sha256 {digest} is not the pinned revision ({PINNED_SHA256})"
        if not args.allow_checksum_mismatch:
            raise SystemExit(f"{problem}; pass --allow-checksum-mismatch to build anyway")
        print(f"warning: {problem}, so the sample will differ")

    # The Hugging Face schema metadata names the labels; check the mapping rather than assume it.
    features = json.loads(pq.read_schema(args.parquet).metadata[b"huggingface"])["info"]["features"]
    names = features["label"]["names"]
    if names != ["neg", "pos"]:
        raise SystemExit(f"unexpected label names {names}; expected ['neg', 'pos']")

    reviews = pd.read_parquet(args.parquet, columns=["text", "label"])
    reviews["text"] = reviews["text"].str.replace(r"<br\s*/?>", " ", regex=True).str.strip()
    # The IMDB train split repeats some reviews; keep one copy so no review lands on both
    # sides of the train/test split that train.py makes.
    unique = reviews.drop_duplicates(subset="text")
    sample = (
        unique.groupby("label", group_keys=False)
        .sample(n=args.per_class, random_state=args.seed)
        .sample(frac=1, random_state=args.seed)  # interleave the two classes
        .reset_index(drop=True)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample[["text", "label"]].to_csv(args.out, index=False, lineterminator="\n")
    counts = sample["label"].value_counts().sort_index().to_dict()
    print(
        f"wrote {len(sample):,} rows to {args.out} from {len(unique):,} unique reviews "
        f"({len(reviews) - len(unique)} duplicates dropped); label counts {counts}"
    )


if __name__ == "__main__":
    main()
