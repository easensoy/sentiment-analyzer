# Sentiment analyzer: end-to-end run report

2026-11-06 · `easensoy/sentiment-analyzer` at `0c3ec53` plus uncommitted changes · CPython 3.15.2 in a uv `.venv`

## Result on the held-out split (2,000 reviews, 1,000 per class)

| | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.9209 | 0.8610 | 0.8899 |
| Positive | 0.8695 | 0.9260 | 0.8969 |
| **Overall** | accuracy **0.8935** | | macro F1 **0.8934** |

The model leans positive. 139 negative reviews were predicted positive, against 74 positive reviews predicted negative.

## What works now

- `python train.py` fits the TF-IDF vectoriser on the 8,000-review training split only, then fits `SVC(kernel="linear")` (42 s, 4,727 support vectors, 59,770 terms). It evaluates on the held-out 20% and saves both fitted objects together in `models/`.
- `python -m uvicorn src.api:app` loads both artefacts at startup. `POST /predict` returns `200 {"sentiment": 0|1}` in about 10–15 ms. If `models/` is missing, startup stops with "run: python train.py".
- `pytest tests/ --cov=src`: **2 passed, 0 failed**, coverage 23% (`api.py` and `model.py` 0%).

## What was missing or broken at `0c3ec53`

**Stopped it running:**
- `data/training_data.csv` was 0 bytes.
- There was no training script.
- The API never trained or loaded anything, so every `/predict` returned **500** "The TF-IDF vectorizer is not fitted". With the vectoriser fitted it was still a 500: "This SVC instance is not fitted yet".
- `save()` stored only the SVC, not the vocabulary.
- On this PC the pins could not install. numpy 1.24.3, pandas 2.0.3 and scikit-learn 1.3.0 have no wheels for Python ≥ 3.12, and httpx 0.24.1 pulls in httpcore 0.17.3, which crashes on import on Python ≥ 3.14.

**Quality only:**
- `tests/test_model.py` and `tests/test_api.py` were empty.
- CI runs Python 3.9 (end-of-life since October 2025) with `actions/*@v2`. It was green only because it tests two preprocessor functions.

**README claims the code did not back up:**
1. "Real-time sentiment analysis": no request could succeed.
2. Output "Positive/Negative": the API returns a bare integer.
3. "Model persistence": the vocabulary was never saved, and nothing called `save` or `load`.
4. "Training interface": there was no data and no script.
5. "Asynchronous processing": the handler is `async def` but runs synchronous, CPU-bound sklearn code, which blocks the event loop.
6. "Error handling": a catch-all 500 that echoes internal messages back to the client.
7. "Input validation": the only check is that `text` is a string; empty or huge input is accepted.
8. "Cleaning and normalization": a single regex that also deletes apostrophes, digits, accents and hyphens (`It's` becomes `its`, `café-style` becomes `cafstyle`).
9. "CI/CD": there is CI only, no deployment.
10. Installation: Unix-only commands, no supported Python version, and no instructions for training or serving.

## Changes (uncommitted)

- `requirements.txt`: numpy → 2.5.4, pandas → 3.1.1, scikit-learn → 1.9.1, httpx → 0.27.2. These were the minimum needed for Python 3.15; the other four pins are unchanged.
- `src/api.py`: a lifespan that loads both artefacts. `/predict` itself is unchanged.
- New `train.py`.
- `.gitignore`: added `.venv/` and `models/`.
- `.github/workflows/python-ci.yml`: now uv-based on Python 3.15 (`actions/checkout@v7`, `astral-sh/setup-uv@v10`), with the same steps as the local setup. A fresh local replay passed (2 tests); it has not yet run on GitHub.
- `scripts/`: `build_dataset.py`, `make_visuals.py`, `docs_screenshot.py`. Their dev-only packages (matplotlib, pyarrow, playwright) are installed in `.venv` only.
- Data: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) at revision `e628166`, train split, with 97 duplicate reviews dropped and a seeded sample of 5,000 negative + 5,000 positive. The licence is "other" (research use; cite Maas et al., 2011), so check it before publishing the CSV.

## Visuals

![Class balance](01_class_balance.png)
![Confusion matrix](02_confusion_matrix.png)
![Precision, recall and F1 per class](03_per_class_metrics.png)
![Top TF-IDF terms by SVC weight](04_top_terms.png)

`top_terms.csv` holds the numbers behind the top-terms chart. Most of those terms are sentiment words. A few are habits of IMDB review writing rather than sentiment (*minutes*, *save*, *today*, *still*).

![FastAPI /docs answering four example reviews](05_fastapi_docs_examples.png)

Example 4, "The movie was not good.", is predicted **positive**. With single-word features, the model sees *good* and misses the *not*. `05a_fastapi_docs_page.png` is one unedited capture of the page.

## Three most valuable next improvements

1. **One sklearn `Pipeline` (TF-IDF with word bigrams, then `LinearSVC`) saved as a single artefact.** One file removes the risk of loading a vectoriser and model that don't match. In a scratch test on the same 2,000 held-out reviews:

   | Model | Accuracy | Fit time | "not good" |
   |---|---|---|---|
   | Current `SVC(kernel="linear")`, unigrams, 8k train | 0.8935 | 42.0 s | positive ✗ |
   | `LinearSVC`, unigrams, 8k train | 0.8965 | 1.3 s | positive ✗ |
   | `LinearSVC`, unigrams + bigrams, 8k train | 0.8930 | 4.4 s | negative ✓ |
   | `LinearSVC`, unigrams + bigrams, all 22.9k unique train reviews | **0.9140** | 12.5 s | negative ✓ |

   "Not bad at all, I really enjoyed it." is still predicted negative by every variant.
2. **Real tests.** CI now runs on Python 3.15, but it still exercises only the two preprocessor tests (23% coverage). Tests worth adding: `clean_text` edge cases, a save/load round-trip, `train.py` on a small fixture CSV, and API tests through `TestClient` (200, 422, startup failure without `models/`).
3. **Modernise the serving layer.** fastapi 0.100 and starlette 0.27 raise 4 deprecation warnings on 3.15, and `asyncio.iscoroutinefunction` is due to be removed in Python 3.16. Upgrade fastapi, uvicorn and pytest. Make the handler a plain `def` so it doesn't block the event loop. Return a label and a score, answer 503 when no model is loaded, and limit input length.
