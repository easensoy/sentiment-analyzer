# Reproducing the study

Every result in the README can be regenerated from the commands below. Commands assume the repository root as the
working directory, Python 3.15 and [uv](https://docs.astral.sh/uv/); on Windows use `.venv\Scripts\python` in place
of `.venv/bin/python`. Downloaded data (about 186 MB) and fitted artefacts stay in git-ignored directories.

## Environment

```bash
uv venv --python 3.15
uv pip install -r requirements-dev.txt    # requirements.txt (train, test, serve) plus matplotlib, pillow, pyarrow, playwright
```

The recorded runs used CPython 3.15.2, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, FastAPI 0.100.0 and uvicorn
0.23.1 (`results/benchmarks/environment.json`). Artefact hashes and fitted parameters are expected to match the
recorded values only under the same library versions.

## Data

```bash
mkdir -p data/raw/fastai-imdb data/raw/hf-imdb-e6281661
curl -L -o data/raw/fastai-imdb/imdb.tgz https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz
curl -L -o data/raw/hf-imdb-e6281661/train-00000-of-00001.parquet \
  https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text/train-00000-of-00001.parquet
curl -L -o data/raw/hf-imdb-e6281661/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text/test-00000-of-00001.parquet
.venv/bin/python -m study.data     # verifies sizes, SHA-256 and cross-source agreement; writes split membership
.venv/bin/python -m study.audit    # exact, normalised and near-duplicate audit; review-id order profile
```

| File | Bytes | SHA-256 |
|---|---:|---|
| `imdb.tgz` (fast.ai mirror of aclImdb v1) | 144,440,600 | `d501018afa17aee9fa1ebe8ac29859a5609980e13dc6e611aa21567cc357351f` |
| `train-00000-of-00001.parquet` | 20,979,968 | `db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c` |
| `test-00000-of-00001.parquet` | 20,470,363 | `b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da` |

`study.data` stops unless both sources contain identical (text, label) multisets in each partition. The canonical
host `ai.stanford.edu` did not accept connections when the study was run; the mirror omits the archive's
`urls_*.txt` movie files, which is why the study approximates movie groups by review-id blocks.

## Historical model

```bash
.venv/bin/python train.py    # internal-holdout protocol of 2026-11-06; publishes models/serving.json (about 70 s)
.venv/bin/python scripts/make_readme_figures.py
```

`train.py` reproduces the original artefact parameter for parameter (`results/historical/refit_equivalence.json`).

## Development selection, fitting and the frozen protocol

```bash
.venv/bin/python -m study.develop --jobs 4               # grouped 5-fold CV on the training partition (about 11 minutes)
.venv/bin/python -m study.develop --svc-reference        # optional: linear-kernel SVC reference folds (slow)
.venv/bin/python -m study.fit                            # the eight frozen candidates (the SVC on 25,000 reviews takes about 17 minutes)
.venv/bin/python -m study.protocol verify                # every locked file must match protocol/LOCK.json
```

The protocol (`protocol/protocol_v1.json`) and its lock (`protocol/LOCK.json`) were pushed before the official test
partition was scored. `study.protocol freeze` rewrites both and must not be run against the published study.

## Official-test evaluation

```bash
.venv/bin/python -m study.evaluate score       # refuses if the lock fails or results/official_test/predictions.csv.gz exists
.venv/bin/python -m study.evaluate analyse     # metrics, paired differences, slices, sensitivity analyses
.venv/bin/python -m study.exploratory          # post hoc class-asymmetry analysis
.venv/bin/python -m study.behavioural evaluate # behavioural suite (provisional; labels unadjudicated)
```

The test partition was scored once. Repeating `score` requires deleting the predictions file first; it should
reproduce the same scores with the same artefacts and library versions.

## Efficiency benchmarks

```bash
.venv/bin/python -m study.benchmark train       # one fresh process per run; 15-30 minutes
.venv/bin/python -m study.benchmark inference   # artefact loading, warm latency, batch throughput; about 10 minutes
.venv/bin/python -m study.benchmark http        # uvicorn on a free loopback port, concurrency sweep; about 12 minutes
.venv/bin/python -m study.benchmark costs       # parameterised compute-cost estimates
```

Benchmarks read only training-partition reviews and do not modify any frozen configuration. On Windows they run on
the performance cores at above-normal priority and record the load of other processes with every measurement; an
otherwise idle machine gives shorter and more stable timings. [`BENCHMARKS.md`](BENCHMARKS.md) documents every
setting.

## Serving

```bash
.venv/bin/python -m uvicorn src.api:app --port 8000    # serves the artefact named by models/serving.json
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "A wonderful, heartfelt film."}'
```

```json
{"label": "positive", "sentiment": 1, "margin": 1.6742724439651835, "model_version": "9e34bdd2005c"}
```

`margin` is the signed decision score, not a probability; the [model card](MODEL_CARD.md#serving-contract) lists the
contract. Any frozen candidate under `models/study/` can be served instead by publishing it to `models/serving.json`
with `src.pipeline.publish`, or by pointing `$SENTIMENT_MODELS_DIR` at a directory with its own `serving.json`, as
the HTTP benchmark does.

## Figures, tables and checks

```bash
.venv/bin/python -m study.efficiency_figures    # figures 09-13 and visual_outputs/study_figures_provenance.json
.venv/bin/python -m study.report                # regenerate every generated table in README.md and docs/
.venv/bin/python -m study.report --check        # CI: fails if any generated table differs from the result files
.venv/bin/python -m pytest tests/
```

## Claude baseline (not executed)

The runner in `study/llm/` evaluates Claude models on the frozen 2,000-review subset
(`protocol/llm_eligible_test_ids.csv`) with prompt version `v1`. It makes network requests only with both `--live`
and `--max-spend-usd`; without them it builds and validates every request and prints a cost bound.

```bash
.venv/bin/python -m study.protocol export-llm   # writes the frozen subset's texts to data/derived/ (not committed)
.venv/bin/python -m study.llm.runner run --examples data/derived/llm_eligible_test.csv \
  --exemplars data/derived/llm_fewshot_exemplars.csv              # dry run
```

See [`study/llm/README.md`](../study/llm/README.md) for truncation, retry, refusal and accounting rules.
