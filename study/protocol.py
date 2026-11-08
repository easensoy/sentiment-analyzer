"""Freeze the evaluation protocol before the official test partition is scored.

    python -m study.protocol freeze        # writes protocol/protocol_v1.json, LLM id lists and protocol/LOCK.json
    python -m study.protocol export-llm    # writes the frozen LLM example and exemplar texts to data/derived/
    python -m study.protocol verify        # exits non-zero unless every locked file is unchanged

The protocol fixes the candidates (configurations and fitted artefact hashes), the development
selection, the paired comparisons, the metrics, the resampling procedure and its sensitivity
analyses, the slice definitions (cut points computed on the training partition only), the
overlap-exclusion sensitivity population, the LLM evaluation subset and few-shot exemplars, and the
behavioural suite version. LOCK.json records the SHA-256 of every file the evaluation depends on;
study/evaluate.py refuses to score unless all of them still match.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import canonical_sha256
from study.configs import candidates, comparisons, selection

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocol"
PROTOCOL = PROTOCOL_DIR / "protocol_v1.json"
LOCK = PROTOCOL_DIR / "LOCK.json"
LLM_IDS = PROTOCOL_DIR / "llm_eligible_test_ids.csv"
LLM_EXEMPLAR_IDS = PROTOCOL_DIR / "llm_fewshot_exemplar_ids.csv"
MEMBERSHIP = ROOT / "results" / "data" / "split_membership.csv.gz"
DERIVED = ROOT / "data" / "derived" / "imdb_official.csv.gz"
FITS = ROOT / "results" / "study" / "fits"
SUITE = ROOT / "behavioural" / "suite_v1.jsonl"
LOCKED_FILES = [
    "protocol/protocol_v1.json", "protocol/llm_eligible_test_ids.csv", "protocol/llm_fewshot_exemplar_ids.csv",
    "results/data/split_membership.csv.gz", "results/data/source_manifest.json", "results/data/duplicate_audit.json",
    "results/data/test_overlap.csv.gz", "results/dev/selection.json", "results/dev/cv_folds.csv",
    "behavioural/suite_v1.jsonl",
    "src/cleaning.py", "src/pipeline.py", "study/configs.py", "study/stats.py", "study/evaluate.py",
    "study/behavioural.py",
]
SEED = 20261108
N_BOOT = 10_000
LLM_PER_CLASS = 1_000


def file_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".py", ".json", ".csv", ".jsonl", ".md"}:  # git stores LF; hash what the repository holds
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def fitted_records() -> dict:
    specs = candidates()
    records = {}
    for candidate_id in specs:
        path = FITS / f"{candidate_id}.json"
        if not path.exists():
            raise SystemExit(f"{candidate_id} is not fitted; run: python -m study.fit")
        records[candidate_id] = json.loads(path.read_text(encoding="utf-8"))
        if records[candidate_id]["config"] != specs[candidate_id]["config"]:
            raise SystemExit(f"{candidate_id}: fitted configuration differs from study/configs.py")
    return records


def llm_subsets(membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    test = membership[membership["partition"] == "test"]
    chosen = [test[test["label"] == label].iloc[np.sort(rng.choice((test["label"] == label).sum(), LLM_PER_CLASS,
                                                                  replace=False))] for label in (0, 1)]
    eligible = pd.concat(chosen)[["example_id", "label"]].sort_values("example_id")
    texts = pd.read_csv(DERIVED, usecols=["example_id", "partition", "label", "text"])
    pool = texts[(texts["partition"] == "train") & texts["text"].str.len().between(300, 700)
                 & ~texts["text"].str.contains("<br", regex=False)]
    exemplars = pd.concat([pool[pool["label"] == label].sample(2, random_state=SEED) for label in (0, 1)])
    return eligible, exemplars[["example_id", "label"]].sort_values("example_id")


def final_model(sel: dict, specs: dict) -> dict:
    """The development-selected final model: best mean CV accuracy among C-tuned full-partition candidates."""
    scored = {}
    for cid, spec in specs.items():
        cfg = spec["config"]
        if spec["training_set"] != "full" or not cfg or cfg["estimator"] not in sel["summary"]:
            continue
        feature = f"{cfg['cleaning']}/{'unigram' if cfg['ngram_range'] == [1, 1] else 'uni+bigram'}"
        scored[cid] = sel["summary"][cfg["estimator"]][feature]["mean_accuracy"]
    best = max(sorted(scored), key=lambda cid: scored[cid])
    return {"rule": "highest mean development accuracy among candidates fitted on the full training partition with "
                    "a development-selected C (test results play no part)",
            "candidate": best, "development_mean_accuracy": scored[best], "considered": scored}


def freeze() -> None:
    sel, specs, comps, fits = selection(), candidates(), comparisons(), fitted_records()
    membership = pd.read_csv(MEMBERSHIP, keep_default_na=False)
    train_words = membership.loc[membership["partition"] == "train", "n_words"]
    cuts = [int(q) for q in np.quantile(train_words, [0.25, 0.5, 0.75])]
    eligible, exemplars = llm_subsets(membership)
    PROTOCOL_DIR.mkdir(exist_ok=True)
    eligible.to_csv(LLM_IDS, index=False, lineterminator="\n")
    exemplars.to_csv(LLM_EXEMPLAR_IDS, index=False, lineterminator="\n")
    if not SUITE.exists():
        raise SystemExit("behavioural/suite_v1.jsonl is missing; run: python -m study.behavioural build")
    protocol = {
        "protocol_version": 1,
        "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "research_question": "On the official IMDB test partition, how do TF-IDF linear classifiers differ in accuracy "
                             "and class-conditional error when estimator, feature representation, cleaning version and "
                             "training-set size are varied one at a time, and how does the historical model compare?",
        "data": {"fitting_and_development": "official training partition (25,000 reviews)",
                 "final_evaluation": "official test partition (25,000 reviews), scored once after this freeze",
                 "membership_file": "results/data/split_membership.csv.gz"},
        "development": {"procedure": sel["rule"], "folds": sel["folds"], "selected_C": sel["selected_C"],
                        "branch_estimator": sel["branch_estimator"]},
        "candidates": {cid: {**specs[cid], "config_sha256": canonical_sha256(specs[cid]["config"]) if specs[cid]["config"] else None,
                             "model_version": fits[cid].get("model_version"), "artefact_sha256": fits[cid].get("artefact_sha256"),
                             "training_ids_sha256": fits[cid]["training_ids_sha256"], "n_train": fits[cid]["n_train"]}
                       for cid in specs},
        "comparisons": comps,
        "final_model": final_model(sel, specs),
        "metrics": {"primary": "accuracy",
                    "secondary": ["macro_f1", "precision_negative", "recall_negative", "error_negative",
                                  "precision_positive", "recall_positive", "error_positive"],
                    "score_policy": "label = positive iff decision score > 0 (ties negative); majority: constant"},
        "statistics": {"resampling": "stratified block bootstrap over review-id order within each (partition, label) "
                                     "directory; blocks resampled with replacement within label strata",
                       "block_ids": 30, "n_boot": N_BOOT, "seed": SEED, "interval": "percentile, 95%",
                       "paired": "one set of block draws shared by all candidates; differences computed per replicate",
                       "sensitivity_block_ids": [1, 60],
                       "multiplicity": f"{len(comps)} pre-specified comparisons; unadjusted 95% intervals plus "
                                       f"Bonferroni-adjusted {100 * (1 - 0.05 / len(comps)):.2f}% intervals",
                       "assumption": "contiguous id blocks approximate movie groups (movie identifiers unavailable)"},
        "slices": {"length_cut_points_words": cuts,
                   "length_definition": "whitespace tokens after replacing <br /> with a space",
                   "length_source": "quartiles of the training partition (frozen here, before test scoring)",
                   "rating": "star rating parsed from the file name: 1-4 negative, 7-10 positive"},
        "overlap_sensitivity": {"exclude_test_reviews": "normalised text occurs in the training partition, or maximum "
                                                        "TF-IDF cosine to a training review >= 0.9",
                                "source": "results/data/test_overlap.csv.gz"},
        "llm_baseline": {"eligible_test_ids": "protocol/llm_eligible_test_ids.csv",
                         "eligible": f"{LLM_PER_CLASS} per class, sampled without replacement (seed {SEED})",
                         "fewshot_exemplar_ids": "protocol/llm_fewshot_exemplar_ids.csv",
                         "exemplars": f"2 per class from the training partition, 300-700 characters, no HTML (seed {SEED})",
                         "status": "runner implemented; no paid execution without an explicit spending limit"},
        "behavioural_suite": {"file": "behavioural/suite_v1.jsonl", "evaluation": "held-out families only",
                              "headline_rule": "unadjudicated cases are excluded from headline claims; results are "
                                               "reported as provisional until a person adjudicates them"},
        "decision_rules": ["No configuration, hyperparameter, cleaning version, slice, metric or resampling choice "
                           "changes after test scoring.",
                           "Any analysis not listed here is reported as exploratory.",
                           "If an evaluation defect is found after scoring, the corrected result is reported "
                           "alongside a description of the defect; the test partition is then no longer described "
                           "as untouched."],
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    write_lock()
    print(f"froze {PROTOCOL.relative_to(ROOT)}: {len(specs)} candidates, {len(comps)} comparisons, "
          f"length cut points {cuts}, LLM subset {len(eligible)}, exemplars {len(exemplars)}")


def write_lock() -> None:
    lock = {"files": {name: file_sha256(ROOT / name) for name in LOCKED_FILES},
            "fits": {p.name: file_sha256(p) for p in sorted(FITS.glob("*.json"))}}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def verify() -> list[str]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    problems = [name for name, digest in lock["files"].items() if file_sha256(ROOT / name) != digest]
    problems += [f"results/study/fits/{name}" for name, digest in lock["fits"].items()
                 if file_sha256(FITS / name) != digest]
    return problems


def export_llm() -> None:
    texts = pd.read_csv(DERIVED, usecols=["example_id", "label", "text"])
    for ids_path, out in ((LLM_IDS, "llm_eligible_test.csv"), (LLM_EXEMPLAR_IDS, "llm_fewshot_exemplars.csv")):
        ids = pd.read_csv(ids_path)
        frame = ids[["example_id"]].merge(texts, on="example_id", validate="one_to_one")
        frame.to_csv(DERIVED.parent / out, index=False, lineterminator="\n")
        print(f"wrote data/derived/{out} ({len(frame)} rows; not committed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["freeze", "verify", "export-llm"])
    command = parser.parse_args().command
    if command == "freeze":
        freeze()
    elif command == "export-llm":
        export_llm()
    else:
        problems = verify()
        print("protocol lock verified" if not problems else f"changed since the freeze: {problems}")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
