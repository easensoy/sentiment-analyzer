# Sentiment analyzer: end-to-end run report

Updated 2026-11-07 · `easensoy/sentiment-analyzer` on `main` · CPython 3.15.2 in a uv `.venv`. This covers the end-to-end run of 2026-11-06 and the day-one fixes that followed an outside review. The [README](../README.md) has the short version.

## Result on the held-out split (2,000 reviews, 1,000 per class)

| | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.9209 | 0.8610 | 0.8899 |
| Positive | 0.8695 | 0.9260 | 0.8969 |
| **Overall** | accuracy **0.8935** (95% CI 0.8792–0.9063) | | macro F1 **0.8934** |

The model leans positive: 139 negative reviews were predicted positive, against 74 positive reviews predicted negative.

The split is random inside IMDB's train set, so reviews of the same movie can sit on both sides. Treat these numbers as optimistic. The interval treats reviews as independent and does not correct for that overlap.

## What works now

- `python train.py` fits the TF-IDF vectoriser on the 8,000-review training split only. It then trains `SVC(kernel="linear")` (about a minute; 4,727 support vectors, 59,770 terms), evaluates on the held-out 20%, and saves both fitted objects plus `metrics.json` to `models/`.
- `python -m uvicorn src.api:app` loads both files at startup and refuses to start without them.
  - `POST /predict` returns `label`, `sentiment`, `margin` (a signed distance, not a probability) and `model_version`.
  - Invalid input gets a 422 with the reason. An unexpected error gets a 500 without internal details.
- 21 tests cover cleaning, the vectoriser, the model, saving and loading, `train.py` end to end, and the HTTP contract. Coverage is 100% of `src/` and 93% of `train.py`. CI runs them on Python 3.15 on every push to `main` and every pull request.

## What was missing or broken at `0c3ec53`

**Stopped it running:**
- `data/training_data.csv` was 0 bytes, and there was no training script.
- The API never trained or loaded anything, so every `/predict` returned 500 ("The TF-IDF vectorizer is not fitted").
- `save()` stored only the SVC, not the vocabulary.
- The 2023 pins could not install on Python 3.12 or later.

**Quality only:**
- `tests/test_model.py` and `tests/test_api.py` were empty.
- CI ran Python 3.9 with `actions/*@v2`.

**README claims the code did not back up:**
1. Real-time analysis: no request could succeed.
2. "Positive/Negative" output: the API returned a bare integer.
3. Model persistence: the vocabulary was never saved.
4. A training interface: there was no data and no script.
5. Asynchronous processing: synchronous, CPU-bound work ran inside `async def`.
6. Error handling: a catch-all 500 that echoed internal messages.
7. Input validation: the only check was that `text` was a string.
8. Cleaning and normalisation: a single regex that also deleted apostrophes, digits, accents and hyphens.
9. CI/CD: there was no deployment.
10. Installation: Unix-only commands, with no Python version and no train or serve steps.

## What changed since `0c3ec53`

- **Python 3.15:**
  - The numpy, pandas, scikit-learn and httpx pins were raised to releases that support it.
  - CI moved from Python 3.9 to 3.15, set up with uv.
- **Data:** a seeded 10,000-review IMDB sample built by `scripts/build_dataset.py`. It uses a pinned Hugging Face revision, drops duplicate reviews, and stops on a checksum mismatch. The Hugging Face card lists the licence as `other` with no further details, so cite Maas et al. (2011) when using the data.
- **Training and serving:** a new `train.py`, and an API that loads the trained files at startup.
- **Fixes after the outside review:**
  - A typed API response with margin and model version, input validation, a plain `def` handler, and no leaked error text.
  - The label now always follows the margin's sign. libsvm's own `predict` broke exact ties the other way; this changed none of the 2,000 held-out predictions.
  - An accuracy interval in `metrics.json`.
  - Tests for the model, the API and training.
  - CI on pull requests, `requirements-dev.txt`, an MIT licence, and a rewritten README.
- **Visuals:** `scripts/make_visuals.py`, and `scripts/docs_screenshot.py`, which drives the installed Edge through Playwright.

## Visuals

![Class balance](01_class_balance.png)
![Confusion matrix](02_confusion_matrix.png)
![Precision, recall and F1 per class](03_per_class_metrics.png)
![Terms with the largest SVC weights](04_top_terms.png)

`top_terms.csv` holds the numbers behind the weights chart. A large weight does not mean a large effect on every review: a term's contribution to one review is its weight times its TF-IDF value in that review. Most of the top terms are sentiment words; a few look like habits of IMDB review writing (*minutes*, *save*, *today*, *still*).

![FastAPI /docs answering four example reviews](05_fastapi_docs_examples.png)

Example 4, "The movie was not good.", is predicted **positive**, with a small margin (+0.30). Single-word features are one plausible cause, but that is a hypothesis, not a tested explanation. `05a_fastapi_docs_page.png` is one unedited capture of the page.

## Next steps

These follow the outside review, and the evaluation work comes before any new modelling.

1. **Protected evaluation:** score on the official IMDB test split, whose movies don't overlap with train. Save per-review predictions and report paired confidence intervals.
2. **Baselines:** majority class, logistic regression and `LinearSVC`. A scratch run, not yet reproducible from this repo, shows what to test:
   - `LinearSVC` scored 0.8965 in 1.3 s, against 0.8935 in 42 s for the current model. That is 6 more correct reviews out of 2,000, too few to call a real difference.
   - Adding bigrams on the same 8,000 reviews scored 0.8930.
   - Bigrams with all 22.9k unique IMDB train reviews reached 0.9140, but that run also used about 2.9× more data, so the gain can't be credited to bigrams.
3. **Behavioural tests and a Claude baseline:** a hand-checked set of negation, contrast, sarcasm and intensity cases, run against the linear models and Claude, with cost.
4. **One versioned `Pipeline`** with a manifest, followed by a model card.
