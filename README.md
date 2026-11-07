# Sentiment Analyzer

[![Python CI](https://github.com/easensoy/sentiment-analyzer/actions/workflows/python-ci.yml/badge.svg)](https://github.com/easensoy/sentiment-analyzer/actions/workflows/python-ci.yml)

A baseline English sentiment classifier: TF-IDF features and a linear SVM, trained on a 10,000-review IMDB sample and served by a FastAPI endpoint. It is evaluated with its weak spots in view, and the roadmap turns it into a comparison of where cheap classifiers and Claude fail.

**Reported result** (2,000 held-out reviews): accuracy **0.8935** (95% CI 0.879–0.906), macro F1 **0.8934**. Recall is 0.861 on negative reviews and 0.926 on positive ones, so the model leans positive.

**Known failure:** "The movie was not good." is classified **positive**, with a margin of +0.30 against about ±3 for clear reviews. Single-word features may be the cause; that hypothesis is not yet tested.

> **Read these numbers with care.** They come from a random 80/20 split inside IMDB's *train* set, where reviews of the same movie can land on both sides. Treat them as optimistic, and don't compare them with results on the standard IMDB test set.

![Confusion matrix: 139 negative reviews were called positive and 74 positive reviews negative](visual_outputs/02_confusion_matrix.png)

## Quickstart

Python 3.15 and [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.15
uv pip install -r requirements.txt
.venv/bin/python train.py                              # about a minute; writes models/
.venv/bin/python -m uvicorn src.api:app --port 8000    # then open http://127.0.0.1:8000/docs
```

On Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text": "A wonderful, heartfelt film."}'
```

## API

`POST /predict` takes `{"text": "..."}` and returns:

```json
{"label": "positive", "sentiment": 1, "margin": 2.7727, "model_version": "98bcdfb288fc"}
```

| Field | Meaning |
|---|---|
| `label` | `negative` or `positive` |
| `sentiment` | The same label as a number: 0 or 1 (kept for existing clients) |
| `margin` | Signed distance from the SVM's decision boundary. Above 0 means positive; exactly 0 counts as negative. A distance, **not a probability**. |
| `model_version` | First 12 hex characters of the SHA-256 of the loaded model files |

`text` must be 1 to 20,000 characters long and contain English letters; anything else gets a 422 with the reason. An unexpected error returns a 500 without internal details. The server refuses to start if `models/` holds no trained model.

## How it works

```text
review ──► clean_text ──────────► TF-IDF ───────────────► linear SVM ──► label + margin
           lowercase ASCII        single words, fitted     scikit-learn
           letters only           on the training split    SVC(kernel="linear")
```

- `src/preprocessor.py` cleans text and holds the TF-IDF vectoriser.
- `src/model.py` wraps the SVM; the label always follows the margin's sign.
- `src/api.py` serves `POST /predict` and loads the model at startup.
- `train.py` splits the data (stratified 80/20, seed 42), fits the vectoriser on the training part only, then trains, evaluates and saves `models/`.

## Evaluation

| Class | Precision | Recall | F1 | Reviews |
|---|---|---|---|---|
| Negative | 0.9209 | 0.8610 | 0.8899 | 1,000 |
| Positive | 0.8695 | 0.9260 | 0.8969 | 1,000 |

Accuracy is 0.8935, with a 95% CI of 0.8792–0.9063. That is a Wilson interval, which treats reviews as independent, so it doesn't account for same-movie overlap. Charts and API screenshots are in [`visual_outputs/`](visual_outputs/), with a written summary in [REPORT.md](visual_outputs/REPORT.md).

| Per-class metrics | Largest SVC weights |
|---|---|
| ![Precision, recall and F1 per class](visual_outputs/03_per_class_metrics.png) | ![Terms with the largest SVC weights toward each class](visual_outputs/04_top_terms.png) |

## Limitations

- **Internal split:** reviews of the same movie can appear in both the training and the evaluation data.
- **Lossy cleaning:** apostrophes, digits, accents and hyphens are deleted (`didn't` becomes `didnt`, `café-style` becomes `cafstyle`). HTML is stripped when the dataset is built, but not from API input.
- **Single words only:** no phrases, so negation and contrast are likely blind spots.
- **Narrow domain:** English movie reviews with strong opinions (IMDB leaves out ratings of 5 and 6 out of 10). Other text is untested.
- **No probabilities:** the margin is not calibrated.
- **Slow to train at scale:** `SVC(kernel="linear")` takes about a minute on 8,000 reviews, and the time grows quickly with more data.

## Data

`data/training_data.csv` holds 10,000 reviews: 5,000 per class, with columns `text` and `label` (0 = negative, 1 = positive). `scripts/build_dataset.py` samples them from the IMDB train split (Maas et al., 2011) through [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) at a pinned revision. It removes duplicate reviews and `<br />` tags, and refuses a file whose checksum doesn't match. To rebuild:

```bash
uv pip install -r requirements-dev.txt
curl -L -o imdb_train.parquet https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text/train-00000-of-00001.parquet
.venv/bin/python scripts/build_dataset.py --parquet imdb_train.parquet
```

**Data terms:** the Hugging Face dataset card lists the licence as `other` and gives no further licensing details. The MIT licence below covers this repository's code, not the review text. If you use the data, cite Maas et al., "Learning Word Vectors for Sentiment Analysis", ACL 2011.

## Development

```bash
uv pip install -r requirements-dev.txt                   # tools for the dataset, charts and screenshots
.venv/bin/python -m pytest tests/ --cov=src --cov=train
.venv/bin/python scripts/make_visuals.py                 # charts from models/ and the data
.venv/bin/python scripts/docs_screenshot.py --url http://127.0.0.1:8000   # needs the API running
```

CI runs the tests on Python 3.15 with uv on every push to `main` and every pull request.

## Roadmap

1. Evaluate on the official IMDB test split, whose movies don't overlap with the train split.
2. Compare against majority-class, logistic-regression and `LinearSVC` baselines, with paired confidence intervals.
3. Save the vectoriser and classifier as one versioned scikit-learn `Pipeline` with a manifest.
4. Build a hand-checked behavioural test set covering negation, contrast, sarcasm and intensity.
5. Run a Claude baseline on the same data and tests, reporting accuracy, cost and latency.
6. Write a model card.

## Licence

Code: [MIT](LICENSE). Data: see [Data terms](#data).
