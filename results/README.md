# Result files

Every file here is produced by a command in the repository and contains no review text. Downloaded
sources and derived text tables stay in the git-ignored `data/raw/` and `data/derived/`.

| Path | Produced by | Contents |
|---|---|---|
| `data/source_manifest.json` | `python -m study.data` | Source URLs, sizes and SHA-256 of the archive and parquet files, cross-source verification, counts, the historical-sample mapping. |
| `data/split_membership.csv.gz` | `python -m study.data` | One row per labelled review: `example_id`, partition, label, review id, star rating, length, 64-bit text hash, bootstrap block, development fold, historical role. |
| `data/duplicate_audit.json` | `python -m study.audit` | Exact and normalised duplicates within and across partitions, the near-duplicate screen, and the review-id order profile. |
| `data/test_overlap.csv.gz` | `python -m study.audit` | Per test review: overlap with the training partition and the highest TF-IDF cosine to a training review. |
| `dev/cv_folds.csv` | `python -m study.develop` | Every development fit: features, cleaning, estimator, C, fold, accuracy and class-conditional errors, fit time. |
| `dev/selection.json` | `python -m study.develop` | Selected C per estimator and feature set, the branch estimator, and the v2 ablation. |
| `dev/cv_svc_reference.csv` | `python -m study.develop --svc-reference` | The linear-kernel SVC at C=1 on the same folds (reference only; it selects nothing). |
| `study/fits/*.json` | `python -m study.fit` | Per frozen candidate: configuration, training-set hash, artefact SHA-256 and fit time. |
| `official_test/` | `python -m study.evaluate score` and `analyse` | Per-review scores and labels for every candidate, metrics, paired differences, slices, sensitivity analyses, inference timing and the scoring log. |
| `behavioural/` | `python -m study.behavioural evaluate` | Per-case predictions and failure rates by phenomenon, split and ambiguity (provisional, unadjudicated). |
| `benchmarks/` | `python -m study.benchmark train`, `inference`, `http`, `costs` | Per-run training time and memory, model-only inference timing with every batch pass, the HTTP concurrency sweep, compute-cost estimates, the workload's review ids (training partition only) and the hardware, software and background-load record; `superseded/` keeps runs replaced by the reported ones. |
| `historical/` | archived from the 2026-11-07 run; `train.py` | The historical model's original metrics, holdout predictions (by row of `data/training_data.csv`) and the record that the Pipeline re-fit equals it. |

`protocol/` holds the frozen evaluation protocol and `LOCK.json`, the SHA-256 of every file the
test-set evaluation depends on; `python -m study.protocol verify` checks them, and CI runs that check.
