"""Fit the frozen candidates on their training sets and record each artefact's identity before any test scoring.

    python -m study.fit                       # every candidate not yet recorded
    python -m study.fit --only svc_uni_full   # one candidate (the SVC fit takes minutes; run it on its own)

Training sets come from results/data/split_membership.csv.gz: "full" is all 25,000 official training
reviews; "historical-8k" is the 8,000 with historical_role == "train". Reviews are fitted in example-id
order. Each fitted Pipeline is published, unserved, to models/study/artefacts/<model_version>/ (not
committed); results/study/fits/<candidate>.json records its configuration, a hash of the training ids,
the artefact SHA-256 and the wall-clock fit time (cleaning + TF-IDF + estimator, one process).

The historical candidate is not re-fitted: it is the artefact train.py publishes (models/serving.json),
whose parameters results/historical/refit_equivalence.json shows to equal the original model's.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import build_pipeline, load, publish
from study.configs import candidates

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived" / "imdb_official.csv.gz"
MEMBERSHIP = ROOT / "results" / "data" / "split_membership.csv.gz"
STUDY_MODELS = ROOT / "models" / "study"
FITS = ROOT / "results" / "study" / "fits"


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def training_frame(training_set: str) -> pd.DataFrame:
    membership = pd.read_csv(MEMBERSHIP, usecols=["example_id", "partition", "historical_role"], keep_default_na=False)
    keep = membership["partition"] == "train"
    if training_set == "historical-8k":
        keep &= membership["historical_role"] == "train"
    ids = set(membership.loc[keep, "example_id"])
    texts = pd.read_csv(DERIVED, usecols=["example_id", "label", "review_id", "text"])
    frame = texts[texts["example_id"].isin(ids)].sort_values("example_id").reset_index(drop=True)
    expected = {"full": 25_000, "historical-8k": 8_000}[training_set]
    if len(frame) != expected:
        raise SystemExit(f"{training_set}: {len(frame)} reviews, expected {expected}")
    return frame


def fit_one(candidate_id: str, spec: dict) -> dict:
    record = {"candidate_id": candidate_id, "role": spec["role"], "config": spec["config"],
              "training_set": spec["training_set"]}
    if candidate_id == "svc_uni_8k_historical":
        pipeline, manifest = load(ROOT / "models")
        if manifest["config"] != spec["config"] or manifest["training"]["rows"] != 8_000:
            raise SystemExit("models/serving.json does not point at the historical artefact; run: python train.py")
        frame = training_frame("historical-8k")
        record.update(model_version=manifest["model_version"], artefact_sha256=manifest["artefact_sha256"],
                      artefact_dir=f"models/artefacts/{manifest['model_version']}",
                      training_ids_sha256=ids_sha256(frame["example_id"]), n_train=8_000,
                      fit_seconds=manifest["training"]["train_seconds"],
                      fit_note="historical artefact published by train.py (fitted on data/training_data.csv rows)")
        return record
    frame = training_frame(spec["training_set"])
    record.update(training_ids_sha256=ids_sha256(frame["example_id"]), n_train=len(frame))
    if spec["config"] is None:  # majority class: ties go to the negative class, as DummyClassifier does
        counts = np.bincount(frame["label"], minlength=2)
        record.update(majority_class=int(np.argmax(counts)), class_counts=counts.tolist(), fit_seconds=0.0)
        return record
    pipeline = build_pipeline(spec["config"])
    start = time.perf_counter()
    pipeline.fit(frame["text"].tolist(), frame["label"].to_numpy())
    seconds = time.perf_counter() - start
    manifest = publish(pipeline, STUDY_MODELS, candidate_id=candidate_id, config=spec["config"], serve=False,
                       training={"training_set": spec["training_set"], "n_train": len(frame),
                                 "training_ids_sha256": record["training_ids_sha256"], "fit_seconds": round(seconds, 2)})
    record.update(model_version=manifest["model_version"], artefact_sha256=manifest["artefact_sha256"],
                  artefact_dir=f"models/study/artefacts/{manifest['model_version']}", n_features=manifest["n_features"],
                  fit_seconds=round(seconds, 2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="candidate ids to fit (default: all not yet recorded)")
    parser.add_argument("--refit", action="store_true", help="fit again even if a record exists")
    args = parser.parse_args()
    specs = candidates()
    FITS.mkdir(parents=True, exist_ok=True)
    for candidate_id in args.only or list(specs):
        path = FITS / f"{candidate_id}.json"
        if path.exists() and not args.refit:
            print(f"{candidate_id}: already recorded")
            continue
        record = fit_one(candidate_id, specs[candidate_id])
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"{candidate_id}: {record.get('model_version', 'rule')} fitted in {record['fit_seconds']} s")


if __name__ == "__main__":
    main()
