# Sentiment Analyzer: where a linear model fails

[![Python CI](https://github.com/easensoy/sentiment-analyzer/actions/workflows/python-ci.yml/badge.svg)](https://github.com/easensoy/sentiment-analyzer/actions/workflows/python-ci.yml)

A TF-IDF + linear SVM sentiment classifier for English movie reviews, built from a 10,000-review IMDB sample, served by FastAPI and analysed for how often it is right and how it goes wrong.

**Accuracy 0.8935** (95% CI 0.879–0.906) and **macro F1 0.8934** on 2,000 held-out reviews, 1,000 per class. The held-out reviews come from a random split *inside* IMDB's train set, not the official test set. Reviews of one movie can sit on both sides, so these numbers are likely optimistic ([protocol](#evaluation-protocol-and-limitations)).

<picture>
  <source media="(min-width: 960px) and (prefers-color-scheme: dark)" srcset="visual_outputs/06_probe_outputs_dark.png">
  <source media="(max-width: 959px) and (prefers-color-scheme: dark)" srcset="visual_outputs/06_probe_outputs_narrow_dark.png" width="360">
  <source media="(max-width: 959px)" srcset="visual_outputs/06_probe_outputs_narrow_light.png" width="360">
  <img src="visual_outputs/06_probe_outputs_light.png" width="830" alt="Table of five hand-written reviews with expected label, predicted label and signed decision score. 'An absolute masterpiece: superb acting and a story I loved from start to finish.': expected positive, predicted positive, +1.12. 'Boring, predictable and badly acted. A total waste of two hours.': expected negative, predicted negative, −3.86. 'I expected to hate it, but it turned out to be surprisingly good.': expected positive, predicted positive, +0.51. 'The movie was good.': expected positive, predicted positive, +0.67. 'The movie was not good.': expected negative, predicted positive, +0.30, wrong.">
</picture>

Five hand-written probes, not dataset reviews: illustrations, not an evaluation. The model gets four right. Adding *not* to “The movie was good.” lowers its score from +0.67 to +0.30, but not below 0, so the label stays positive. Data: [`06_probe_outputs.csv`](visual_outputs/06_probe_outputs.csv).

[Reproduce](#reproduce) · [Evaluation protocol](#evaluation-protocol-and-limitations) · [Error analysis](#error-analysis) · [API](#api) · [Code](src/) · [Run report](visual_outputs/REPORT.md)

## Main findings

Each finding uses the 2,000 held-out reviews, except where it names a probe.

1. **It leans positive.** It misclassifies 13.9% of negative reviews (139 of 1,000; 95% CI 11.9–16.2%), against 7.4% of positive ones (74 of 1,000; 5.9–9.2%).
2. **The lean appears in all four length quarters** (exploratory). Negative reviews have the higher error rate in each quarter, and neither class shows a clear trend with length ([figure](#errors-by-review-length)).
3. **It reads *not* as just another word.** In the probe “The movie was not good.”, the term *not* adds −0.29 to the score and *good* adds +0.85. With *was*, *the* and *movie* (−0.26 together), the score is +0.30, so the sentence is labelled positive ([figure](#one-wrong-prediction-decomposed)). One probe shows the mechanism; it does not measure how often negation fails.

## Reproduce

Python 3.15 and [uv](https://docs.astral.sh/uv/). On Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`.

```bash
uv venv --python 3.15
uv pip install -r requirements-dev.txt               # includes requirements.txt
.venv/bin/python train.py                            # about a minute; writes models/
.venv/bin/python scripts/make_readme_figures.py      # checks models/, then draws figures 06-08
.venv/bin/python -m pytest tests/
.venv/bin/python -m uvicorn src.api:app --port 8000  # then open http://127.0.0.1:8000/docs
```

`requirements.txt` alone is enough to train, test and serve; the dev file adds the chart, dataset and screenshot tools. `train.py` fits the vectoriser on the training split only, then saves the model, the vectoriser, `metrics.json` and the held-out predictions to `models/`. The figure script stops unless the seeded split rebuilds the saved held-out set row for row, the saved model reproduces all 2,000 saved predictions, and the recomputed accuracy and confusion matrix equal `metrics.json`.

## Evaluation protocol and limitations

- **Data:** 10,000 reviews, 5,000 per class, sampled with seed 42 from the 25,000-review IMDB *train* split (Maas et al., 2011) after removing duplicate reviews.
- **Split:** stratified and random, 80/20 with seed 42: 8,000 training reviews and 2,000 held out. The TF-IDF vocabulary (59,770 single words) and its IDF weights come from the training reviews only.
- **Model:** `SVC(kernel="linear")` with 4,727 support vectors. The label follows the signed decision score: above 0 is positive, 0 or below is negative.
- **Intervals:** 95% Wilson score intervals. They treat held-out reviews as independent and take this fitted model and split as fixed, so they cover the sampling of held-out reviews only, not variation across splits or training runs. The eight intervals in the length analysis are not adjusted for multiple comparisons.

| Class | Precision | Recall | F1 | Reviews |
|---|---|---|---|---|
| Negative | 0.921 | 0.861 | 0.890 | 1,000 |
| Positive | 0.869 | 0.926 | 0.897 | 1,000 |

What limits these numbers:

- **Same-movie overlap.** IMDB's official train and test splits contain different movies, with up to 30 reviews per movie. This split is drawn from inside the train split, so one movie's reviews can land on both sides, along with its names and plot words. The accuracy is likely optimistic, and the intervals are likely too narrow because they ignore this grouping.
- **Not a benchmark result.** Published IMDB results usually use the official test set and the full training set, so don't compare these numbers with them.
- **Single words only.** The features carry no word order, so negation and contrast reach the model only as separate words.
- **Lossy cleaning.** Apostrophes, digits, hyphens and accented letters are deleted (`didn't` becomes `didnt`, `café-style` becomes `cafstyle`), and one-letter words are ignored. The dataset build replaces `<br />` tags; API input gets no HTML handling.
- **Narrow domain.** English movie reviews with clear opinions: IMDB's labelled reviews leave out ratings of 5 and 6 out of 10. Other text is untested.
- **No probabilities.** The signed decision score is uncalibrated, and its scale is specific to this model.
- **Slow to train at scale.** `SVC` took 67 s to train on 8,000 reviews here, and scikit-learn's documentation notes that its fit time grows at least quadratically with the number of samples.

## Error analysis

### Errors by review length

<picture>
  <source media="(min-width: 960px) and (prefers-color-scheme: dark)" srcset="visual_outputs/07_error_by_length_dark.png">
  <source media="(max-width: 959px) and (prefers-color-scheme: dark)" srcset="visual_outputs/07_error_by_length_narrow_dark.png" width="360">
  <source media="(max-width: 959px)" srcset="visual_outputs/07_error_by_length_narrow_light.png" width="360">
  <img src="visual_outputs/07_error_by_length_light.png" width="830" alt="Error rate with 95% intervals by review-length quarter and true class. Negative reviews: 15.3% for 26 to 125 words, 11.8% for 126 to 170, 15.4% for 171 to 282, 13.4% for 283 to 1,269, and 13.9% overall. Positive reviews: 7.3%, 5.6%, 9.5% and 6.9%, and 7.4% overall. Negative reviews are higher in every quarter, with no clear trend by length.">
</picture>

The quarters split the 2,000 held-out reviews by word count (whitespace tokens, both classes pooled), with cut points at 125, 170 and 282 words. Each dot is the share of one class's reviews that the model got wrong. Its 95% Wilson interval covers that cell alone, treats reviews as independent and is not adjusted for the eight comparisons. The direction is the same in all four quarters. As an exploratory association, it does not say why the model leans positive. Data: [`07_error_by_length.csv`](visual_outputs/07_error_by_length.csv).

### One wrong prediction, decomposed

<picture>
  <source media="(min-width: 960px) and (prefers-color-scheme: dark)" srcset="visual_outputs/08_score_contributions_dark.png">
  <source media="(max-width: 959px) and (prefers-color-scheme: dark)" srcset="visual_outputs/08_score_contributions_narrow_dark.png" width="360">
  <source media="(max-width: 959px)" srcset="visual_outputs/08_score_contributions_narrow_light.png" width="360">
  <img src="visual_outputs/08_score_contributions_light.png" width="830" alt="Waterfall of the contributions to the score of 'The movie was not good.': good +0.850, not −0.295, was −0.270, the +0.152, movie −0.138, no other terms, intercept +0.002. They sum to +0.301, which is above 0, so the label is positive.">
</picture>

A linear SVM's score is a sum, so this prediction decomposes exactly. Each term contributes its SVM weight times its TF-IDF value in this text, and the intercept is added. The five terms and the intercept add up to the model's signed decision score of +0.3006; the script checks the match to within 1e-9. Here *not* is a single feature with a fixed weight of −0.66, whatever word it negates. Its contribution changes only through the unit-length scaling of each review's TF-IDF vector, never through word order, and in this sentence it is about a third the size of *good*'s. Without *not*, “The movie was good.” scores +0.665. The drop of 0.36 is *not*'s own −0.29 plus about 0.07 from rescaling the other words. These are contributions within this model, not causal explanations of language. Data: [`08_score_contributions.csv`](visual_outputs/08_score_contributions.csv).

## API

`POST /predict` takes `{"text": "..."}`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A wonderful, heartfelt film."}'
```

```json
{"label": "positive", "sentiment": 1, "margin": 1.6742724439651835, "model_version": "98bcdfb288fc"}
```

| Field | Meaning |
|---|---|
| `label` | `negative` or `positive` |
| `sentiment` | The same label as 0 or 1, kept for existing clients |
| `margin` | The signed decision score at full precision: above 0 is positive, 0 or below is negative. Despite the field name, it is neither a distance nor a probability. |
| `model_version` | First 12 hex characters of the SHA-256 of the loaded model files |

The API returns 422 with the reason when `text` is empty, longer than 20,000 characters or has no English letters. It also returns 422 (type `no_recognized_vocabulary`) when none of the words are in the model's vocabulary, because then there is nothing to score. Unexpected errors return 500 without internal details, and the server refuses to start without a trained model. [FastAPI `/docs` screenshots](visual_outputs/REPORT.md#visuals) show four of the probes answered through the API.

## Supporting results and provenance

- **[Run report](visual_outputs/REPORT.md):** what was fixed, the full metrics, the tables behind each figure and the `/docs` screenshots.
- **Figure data:** [`06_probe_outputs.csv`](visual_outputs/06_probe_outputs.csv), [`07_error_by_length.csv`](visual_outputs/07_error_by_length.csv) and [`08_score_contributions.csv`](visual_outputs/08_score_contributions.csv). [`figures_provenance.json`](visual_outputs/figures_provenance.json) records the commit the figures were made from, the SHA-256 of every input, the model version, the checks that passed and the library versions.

<details>
<summary>Confusion matrix, class balance, per-class metrics and the largest SVM weights</summary>

| Confusion matrix | Class balance |
|---|---|
| ![Confusion matrix: of 1,000 negative reviews, 861 predicted negative and 139 positive; of 1,000 positive reviews, 74 predicted negative and 926 positive](visual_outputs/02_confusion_matrix.png) | ![Class balance: 5,000 reviews per class in the sample, 4,000 per class in training and 1,000 per class held out](visual_outputs/01_class_balance.png) |
| **Per-class metrics** | **Largest SVM weights** |
| ![Precision, recall and F1 per class on the held-out split](visual_outputs/03_per_class_metrics.png) | ![The 20 terms with the largest SVM weights toward each class, led by worst and great](visual_outputs/04_top_terms.png) |

Generated by `scripts/make_visuals.py`; the weights are in [`top_terms.csv`](visual_outputs/top_terms.csv). A large weight does not mean a large contribution to every review: a term's contribution to one review is its weight times its TF-IDF value there.

</details>

### Data

`data/training_data.csv` holds the 10,000 reviews, with columns `text` and `label` (0 = negative, 1 = positive). `scripts/build_dataset.py` samples them from the IMDB train split through [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) at a pinned revision. It drops duplicate reviews, replaces `<br />` tags and refuses a source file whose checksum doesn't match. To rebuild:

```bash
curl -L -o imdb_train.parquet https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text/train-00000-of-00001.parquet
.venv/bin/python scripts/build_dataset.py --parquet imdb_train.parquet
```

**Data terms:** the Hugging Face dataset card lists the licence as `other` and gives no further details. The MIT licence below covers this repository's code, not the review text. If you use the data, cite Maas et al., "Learning Word Vectors for Sentiment Analysis", ACL 2011.

## Roadmap

None of this exists in the repository yet. The next steps will:

1. evaluate on the official IMDB test split, whose movies don't overlap with the train split;
2. compare against majority-class, logistic-regression and `LinearSVC` baselines on the same reviews, with paired confidence intervals;
3. build a hand-checked test set for negation, contrast, sarcasm and intensity, so that failures like the one above are counted rather than illustrated;
4. run Claude on the same held-out reviews and the hand-checked set, reporting accuracy, cost and latency;
5. save the vectoriser and classifier as one versioned scikit-learn `Pipeline`, with a model card.

## Licence

Code: [MIT](LICENSE). Data: see [Data terms](#data).
