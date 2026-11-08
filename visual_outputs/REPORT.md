# Sentiment analyzer: end-to-end run report

Updated 2026-11-08 · `easensoy/sentiment-analyzer` on `main` · CPython 3.15.2 in a uv `.venv`. This covers the end-to-end run of 2026-11-06, the fixes that followed an outside review, and the error analysis behind the README figures. The [README](../README.md) has the short version.

## Result on the held-out split (2,000 reviews, 1,000 per class)

| | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.9209 | 0.8610 | 0.8899 |
| Positive | 0.8695 | 0.9260 | 0.8969 |
| **Overall** | accuracy **0.8935** (95% CI 0.8792–0.9063) | | macro F1 **0.8934** |

The model leans positive: 139 of 1,000 negative reviews were predicted positive (13.9%, 95% CI 11.9–16.2%), against 74 of 1,000 positive reviews predicted negative (7.4%, 5.9–9.2%).

The held-out reviews are a stratified random 20% (seed 42) of a 10,000-review sample of IMDB's train split, so reviews of the same movie can sit on both sides, and these numbers are likely optimistic. All intervals here are 95% Wilson score intervals. They treat reviews as independent and take this fitted model and split as fixed, so they cover the sampling of held-out reviews only and do not correct for the same-movie overlap.

## What works now

- `python train.py` fits the TF-IDF vectoriser on the 8,000-review training split only. It then trains `SVC(kernel="linear")` (67 s on this machine; 4,727 support vectors, 59,770 terms), evaluates on the held-out 20%, and saves both fitted objects, `metrics.json` and the held-out predictions to `models/`.
- `python -m uvicorn src.api:app` loads both files at startup and refuses to start without them.
  - `POST /predict` returns `label`, `sentiment`, `margin` (the linear SVM's signed decision score at full precision, not a probability) and `model_version`. The label comes from that same score: above 0 is positive, 0 or below is negative.
  - Invalid input gets a 422 with the reason. That includes text with no word from the model's vocabulary (a single letter, or only unseen words). An unexpected error gets a 500 without internal details.
- `python scripts/make_readme_figures.py` re-derives the README figures from `models/` and the data. It stops unless the seeded split rebuilds the saved held-out set row for row, the saved model reproduces all 2,000 saved predictions, the recomputed accuracy and confusion matrix equal `metrics.json`, and the decomposed score adds up to the model's own score.
- 30 tests cover cleaning, the vectoriser, the model and its score-to-label policy, saving and loading, `train.py` end to end, and the HTTP contract. CI runs them on Python 3.15 on every push to `main` and every pull request. The tests check behaviour, not model quality.

## Error analysis behind the README figures

The figures come from `scripts/make_readme_figures.py`, which writes each one in a light and a dark version (figure 06 also in a narrow layout for phone screens), the numbers behind it as CSV, and [`figures_provenance.json`](figures_provenance.json). That file records the commit the figures were made from, the SHA-256 of every input, the model version (`98bcdfb288fc`), the checks that passed and the library versions.

### Hand-written probes (figure 06)

<picture>
  <source media="(min-width: 960px) and (prefers-color-scheme: dark)" srcset="06_probe_outputs_dark.png">
  <source media="(max-width: 959px) and (prefers-color-scheme: dark)" srcset="06_probe_outputs_narrow_dark.png" width="360">
  <source media="(max-width: 959px)" srcset="06_probe_outputs_narrow_light.png" width="360">
  <img src="06_probe_outputs_light.png" width="830" alt="Five hand-written reviews with expected label, predicted label and signed decision score; only 'The movie was not good.' is wrong, scored +0.30.">
</picture>

| # | Probe | Expected | Predicted | Signed decision score |
|---|---|---|---|---|
| 1 | An absolute masterpiece: superb acting and a story I loved from start to finish. | positive | positive | +1.1190 |
| 2 | Boring, predictable and badly acted. A total waste of two hours. | negative | negative | −3.8567 |
| 3 | I expected to hate it, but it turned out to be surprisingly good. | positive | positive | +0.5134 |
| 4 | The movie was good. | positive | positive | +0.6650 |
| 5 | The movie was not good. | negative | **positive** | +0.3006 |

These are illustrations, not an evaluation: five sentences written by hand, where "expected" is the sentiment each was written to express. Probes 1, 2, 3 and 5 are the `/docs` screenshot examples, and the API returned the same scores there. Probe 4 is probe 5 without *not*.

### Error rate by review length (figure 07)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="07_error_by_length_dark.png">
  <img src="07_error_by_length_light.png" width="830" alt="Error rate by length quarter and true class with 95% intervals; negative reviews are higher in every quarter, with no clear length trend.">
</picture>

| Review length | Negative reviews wrong | Positive reviews wrong |
|---|---|---|
| 26–125 words | 36 / 235 = 15.3% (11.3–20.5%) | 20 / 273 = 7.3% (4.8–11.0%) |
| 126–170 words | 33 / 280 = 11.8% (8.5–16.1%) | 12 / 213 = 5.6% (3.3–9.6%) |
| 171–282 words | 38 / 246 = 15.4% (11.5–20.5%) | 24 / 253 = 9.5% (6.5–13.7%) |
| 283–1,269 words | 32 / 239 = 13.4% (9.6–18.3%) | 18 / 261 = 6.9% (4.4–10.6%) |
| All lengths | 139 / 1,000 = 13.9% (11.9–16.2%) | 74 / 1,000 = 7.4% (5.9–9.2%) |

Method: word count is the number of whitespace-separated tokens in the stored review text. The quarters are quartiles of that count over all 2,000 held-out reviews, with both classes pooled (`pandas.qcut`), so each quarter holds about 500 reviews but not 250 per class. Intervals are 95% Wilson intervals per cell, treating reviews as independent, and are not adjusted for the eight comparisons.

Reading: negative reviews have the higher error rate in every quarter, and neither class moves steadily with length. This is an exploratory association on one internal split. It suggests that the positive lean is not confined to short or long reviews, but it says nothing about what causes the lean.

### One score, decomposed (figure 08)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="08_score_contributions_dark.png">
  <img src="08_score_contributions_light.png" width="830" alt="Contributions to the score of 'The movie was not good.': good +0.850, not −0.295, was −0.270, the +0.152, movie −0.138, intercept +0.002, total +0.301.">
</picture>

| Component | SVM weight | TF-IDF value | Contribution | Running total |
|---|---|---|---|---|
| good | +1.4565 | 0.5834 | +0.8497 | +0.8497 |
| not | −0.6602 | 0.4463 | −0.2946 | +0.5551 |
| was | −0.6413 | 0.4203 | −0.2696 | +0.2855 |
| the | +0.5118 | 0.2963 | +0.1516 | +0.4372 |
| movie | −0.3119 | 0.4427 | −0.1381 | +0.2991 |
| other terms (none) | | | 0 | +0.2991 |
| intercept | | | +0.0015 | +0.3006 |
| **signed decision score** | | | **+0.3006** | |

The SVM is linear, so its signed decision score is the intercept plus each term's weight times its TF-IDF value. The script stops if these parts differ from `decision_function` by more than 1e-9; the actual difference is about 3e-14. In this model *not* is one unigram feature with a fixed weight of −0.66, whatever word it negates. Its contribution changes only through the unit-length scaling of each review's TF-IDF vector, never through word order, so the model cannot represent what *not* applies to. Here it removes about a third of what *good* adds. Without *not*, the sentence scores +0.6650: dropping a term also rescales the other TF-IDF values, because each review's vector is normalised to unit length.

This explains the score within this model, not how people read negation. Whether phrase features such as bigrams would fix negation in general is untested.

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
  - A typed API response with the signed decision score (`margin`) and model version, input validation, a plain `def` handler, and no leaked error text.
  - An explicit application policy for turning the score into a label: above 0 is positive; 0 or below is negative. scikit-learn's `SVC.predict` follows libsvm's own convention for an exact tie, which need not match this policy, so the label is derived from the score and always agrees with it. Adopting the policy changed none of the 2,000 held-out predictions.
  - The score is computed once per request and returned at full precision, so a tiny positive score can't come back as `0.0` next to a positive label. Text with no recognized vocabulary gets a 422, instead of a label decided by the intercept alone.
  - An accuracy interval in `metrics.json`.
  - Tests for the model, the API and training.
  - CI on pull requests, `requirements-dev.txt`, an MIT licence, and a rewritten README.
- **Visuals:** `scripts/make_visuals.py` for the supporting charts, `scripts/docs_screenshot.py`, which drives the installed Edge through Playwright, and `scripts/make_readme_figures.py` for the README figures and their checks.

## Visuals

These supporting charts come from `scripts/make_visuals.py` and are drawn for a light background.

![Class balance](01_class_balance.png)
![Confusion matrix](02_confusion_matrix.png)
![Precision, recall and F1 per class](03_per_class_metrics.png)
![Terms with the largest SVC weights](04_top_terms.png)

`top_terms.csv` holds the numbers behind the weights chart. A large weight does not mean a large contribution to every review: a term's contribution to one review is its weight times its TF-IDF value in that review (figure 08 shows one review). Most of the top terms are sentiment words. A few (*minutes*, *save*, *today*, *still*) look like habits of IMDB review writing; that is an impression, not a tested finding.

![FastAPI /docs answering four example reviews](05_fastapi_docs_examples.png)

Example 4, "The movie was not good.", is predicted **positive**, with a small signed decision score (+0.30, rounded for display); figure 08 shows how that score is made up. `05a_fastapi_docs_page.png` is one unedited capture of the page.

## Next steps

None of this exists in the repository yet. The evaluation work comes before any new modelling, and the next steps will:

1. score the model on the official IMDB test split, whose movies don't overlap with train, saving per-review predictions;
2. add majority-class, logistic-regression and `LinearSVC` baselines, with and without bigrams, trained on the same reviews so that differences in data size can't explain differences in accuracy, and report paired confidence intervals for the differences;
3. build a hand-checked set of negation, contrast, sarcasm and intensity cases, and run the linear models and Claude on it and on the same held-out reviews, reporting accuracy, cost and latency;
4. save one versioned `Pipeline` with a manifest, followed by a model card.
