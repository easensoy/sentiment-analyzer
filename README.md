# Accuracy and compute cost of TF-IDF linear sentiment classifiers on the official IMDB test partition

[![Python CI](https://github.com/easensoy/sentiment-analyzer/actions/workflows/python-ci.yml/badge.svg)](https://github.com/easensoy/sentiment-analyzer/actions/workflows/python-ci.yml)

**Research question.** On the official IMDB test partition (Maas et al., 2011), how do TF-IDF linear classifiers
differ in accuracy and class-conditional error when the estimator, the feature representation, the text-cleaning
version and the training-set size are changed one at a time, and what does each configuration cost to train and to
serve?

Eight configurations, including the repository's historical model and a majority-class rule, were fixed in a
protocol that was pushed before the 25,000 official test reviews were scored once. Training and serving costs were
measured afterwards on the frozen artefacts, single-threaded, on one laptop CPU.

[Findings](#findings) · [Results](#results) · [Design](#implementation-and-design) · [Training, serving and
cost](#training-serving-and-cost) · [Behavioural robustness](#behavioural-robustness-provisional) ·
[Limitations](#limitations-and-validity-threats) · [Documentation](#documentation)

## Findings

Timings are single-threaded on a performance core of an Intel Core Ultra 7 165H under Windows 11, measured while
other processes shared the host ([conditions](docs/BENCHMARKS.md#environment-and-conditions)). Accuracy intervals are
95% block-bootstrap intervals on the 25,000 official test reviews; cost ranges span repeated passes.

1. **The served historical model is dominated on accuracy and on compute.** `svc_uni_8k_historical` scores 86.85%
   [86.32–87.37] and needs 5.30 CPU seconds per 1,000 predictions, measured in process on 2,000 training-partition
   reviews in batches of 1,024 (median of five passes). The final model, `linsvc_bi_v2_full`, scores 90.26%
   [89.80–90.70], +3.42 pp [+3.05, +3.78] in the paired comparison, at 0.492 s; the historical model needed 10.1–11.5
   times as much CPU in each pass. The historical model is a linear-kernel `SVC`, which libsvm scores by evaluating
   the kernel against each of its 4,727 support vectors; the LinearSVC models compute one sparse dot product.
2. **Word bigrams buy accuracy at a moderate cost; cleaning v2 adds cost without measurable accuracy.** In the
   pre-specified LinearSVC comparisons, 1–2-grams add +1.66 pp [+1.39, +1.93] over unigrams for 2.4 times the batch
   CPU per prediction (0.320 against 0.133 s per 1,000), 3.6 times the fit time and 3.9 times the fit's memory
   increment. Cleaning v2, part of the final model, changes accuracy by −0.01 pp [−0.18, +0.16] for 1.48–1.60 times
   the batch CPU of `linsvc_bi_full` in each pass and 1.31 times its fit time, so `linsvc_bi_full` dominates the final
   model on point estimates. The final model was selected on development accuracy before any cost was measured, and
   this comparison is post hoc.
3. **Over HTTP, request handling costs more CPU than the linear models' own scoring.** Behind one uvicorn worker on
   loopback, the final model's server uses 2.1 ms of CPU per request at its peak throughput (2.1–2.5 ms over all
   concurrency levels and sweeps), against an in-process single-review p50 of 0.90 ms and 0.49 ms of CPU per
   prediction in batches. One worker saturates about one core at 544 requests/s (sweeps 540–549, concurrency 16, p95
   34.5 ms); the historical SVC, whose own scoring dominates at 8.7 ms of server CPU per request, saturates at 117
   requests/s. None of the 203,384 requests failed.

<picture>
  <source media="(min-width: 960px) and (prefers-color-scheme: dark)" srcset="visual_outputs/11_accuracy_vs_compute_dark.png">
  <source media="(max-width: 959px) and (prefers-color-scheme: dark)" srcset="visual_outputs/11_accuracy_vs_compute_narrow_dark.png" width="360">
  <source media="(max-width: 959px)" srcset="visual_outputs/11_accuracy_vs_compute_narrow_light.png" width="360">
  <img src="visual_outputs/11_accuracy_vs_compute_light.png" width="830" alt="Scatter plot of official-test accuracy against CPU seconds per 1,000 predictions on a log scale, with 95% accuracy intervals and pass ranges: LinearSVC, unigrams, 8k 86.80% at 0.117 s; logistic regression 88.71% at 0.125 s; LinearSVC, unigrams, 25k 88.61% at 0.133 s; LinearSVC, 1–2-grams 90.27% at 0.320 s; final model, 1–2-grams with cleaning v2, 90.26% at 0.492 s; historical SVC 86.85% at 5.30 s; SVC, 25k 88.39% at 12.1 s. The dashed Pareto line joins LinearSVC 8k, logistic regression and LinearSVC with 1–2-grams.">
</picture>

Accuracy comes from the single scoring run on the 25,000 official test reviews and CPU time from 2,000
training-partition reviews, so each axis uses one population for every configuration. The three unigram LinearSVC and
logistic-regression configurations cost the same to within the process clock's resolution, so the left end of the post
hoc Pareto set is not resolved. Two comparisons are resolved: each SVC is less accurate than `linsvc_bi_full`, with
non-overlapping intervals, at 17 and 38 times its CPU time; and `linsvc_bi_full` matches the final model's accuracy at
0.65 times its CPU time. Phone-width layout: [narrow figures](docs/narrow-figures.md).

## Results

Accuracy and class-conditional error on the 25,000 official test reviews, from the single scoring run, with 95%
block-bootstrap intervals. Fit: complete `Pipeline.fit` in seconds, median of three runs in fresh processes (one for
`svc_uni_full`). Fit memory: increase of the process's peak working set during the fit. CPU s per 1,000 predictions:
in-process batch scoring at batch size 1,024, including cleaning and TF-IDF. p50 / p95: single-review requests in
process, without HTTP. Timings are single-threaded on the performance cores of an Intel Core Ultra 7 165H with
other processes running ([conditions](docs/BENCHMARKS.md#environment-and-conditions)). Bold: Pareto-efficient in
accuracy versus CPU time, computed post hoc on point estimates. Macro F1, paired comparisons and sensitivity
analyses: [supplementary results](docs/RESULTS.md).

<!-- table:summary -->
| Configuration | Model | Accuracy % [95% CI] | Error, negative / positive % | Fit, s | Fit memory, MB | CPU s per 1,000 predictions | Single-request p50 / p95, ms |
|---|---|---|---|---:|---:|---:|---|
| `majority` | majority class | 50.00 [49.94–50.06] | 0.00 / 100.00 | — | — | — | — |
| `svc_uni_8k_historical` | SVC, unigrams, legacy-v1, C=1, 8k | 86.85 [86.32–87.37] | 13.32 / 12.98 | 41.2 | 222 | 5.30 | 7.41 / 10.41 |
| `svc_uni_full` | SVC, unigrams, legacy-v1, C=1, 25k | 88.39 [87.90–88.88] | 11.06 / 12.16 | 641 | 415 | 12.1 | 16.65 / 23.21 |
| **`logreg_uni_full`** | LogReg, unigrams, legacy-v1, C=2, 25k | 88.71 [88.21–89.20] | 11.14 / 11.45 | 4.00 | 159 | 0.125 | 0.57 / 0.95 |
| `linsvc_uni_full` | LinearSVC, unigrams, legacy-v1, C=0.125, 25k | 88.61 [88.11–89.10] | 11.44 / 11.34 | 3.30 | 154 | 0.133 | 0.59 / 0.95 |
| **`linsvc_uni_8k`** | LinearSVC, unigrams, legacy-v1, C=0.125, 8k | 86.80 [86.26–87.33] | 14.33 / 12.06 | 1.07 | 56 | 0.117 | 0.55 / 0.93 |
| **`linsvc_bi_full`** | LinearSVC, 1–2-grams, legacy-v1, C=1, 25k | 90.27 [89.80–90.72] | 9.94 / 9.52 | 12.0 | 602 | 0.320 | 0.71 / 1.38 |
| `linsvc_bi_v2_full` | LinearSVC, 1–2-grams, v2, C=0.5, 25k | 90.26 [89.80–90.70] | 10.14 / 9.34 | 15.8 | 593 | 0.492 | 0.90 / 1.94 |
<!-- /table:summary -->

The historical model's original evaluation, on an internal holdout that did not separate movies between training and
evaluation, reported 89.35% (Wilson interval 87.92–90.63%), 2.50 points above its official-test accuracy. For the
final model, errors concentrate near the rating boundary: 21.4% of reviews rated 4 and 18.9% of those rated 7 are
misclassified, against 4.7% at rating 1 and 6.0% at rating 10. Removing the 124 test reviews that overlap the training
partition changes no trained configuration's accuracy by more than 0.05 points.

## Implementation and design

| Component | Specification |
|---|---|
| Data | Large Movie Review Dataset v1.0: official training and test partitions, 25,000 reviews each, 12,500 per class, star ratings 1–4 (negative) and 7–10 (positive). Sources: the fast.ai mirror of the aclImdb archive and the Hugging Face `stanfordnlp/imdb` parquet files at revision `e6281661`, pinned by SHA-256 and verified to contain identical (text, label) multisets. |
| Models | One scikit-learn `Pipeline` per configuration: cleaning (`legacy-v1` or `v2`, `src/cleaning.py`), `TfidfVectorizer` over word unigrams or 1–2-grams (scikit-learn defaults, `min_df=1`), then `SVC(kernel="linear")`, `LinearSVC` or `LogisticRegression` (liblinear). Each artefact carries a manifest with its configuration, training-set hash, SHA-256 and library versions, and is validated when loaded. |
| Development | Official training partition only: five folds of 100-id blocks per label, the vectoriser refitted in each training fold, C chosen by mean validation accuracy. The final model, `linsvc_bi_v2_full`, was named from development accuracy alone. |
| Evaluation | One scoring of the official test partition after the protocol freeze. 95% intervals from a stratified bootstrap of 30-id blocks (10,000 replicates), which approximates resampling by movie; paired differences share the replicates; Bonferroni adjustment for the seven pre-specified comparisons. |
| Efficiency benchmark | Added after scoring; frozen configurations only. Workload: 2,000 official training-partition reviews. Numerical libraries pinned to one thread; benchmark processes on the performance cores at above-normal priority; each training run in a fresh process; HTTP served by uvicorn with one worker over loopback, under a closed-loop concurrency sweep. Boundaries and conditions: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). |
| Serving | FastAPI `POST /predict` returns the label, the signed decision score and the artefact identifier; the server refuses an artefact whose manifest, SHA-256, structure or scikit-learn version does not match ([model card](docs/MODEL_CARD.md#serving-contract)). |

**Provenance.** The protocol ([`protocol_v1.json`](protocol/protocol_v1.json)) and the SHA-256 lock of every file the
evaluation depends on ([`LOCK.json`](protocol/LOCK.json)) were pushed up to commit
[`88d8102`](https://github.com/easensoy/sentiment-analyzer/commit/88d810245a81e985f03b6ad343e309532a9c15b1), whose CI
run started at 19:38:22 UTC on 2026-11-08 ([run
42516034238](https://github.com/easensoy/sentiment-analyzer/actions/runs/42516034238)); scoring started at 19:38:39
UTC. The scoring log ([`scoring_log.json`](results/official_test/scoring_log.json)) records `88d8102` with uncommitted
changes confined to presentation files outside the lock. The repository's history has since been rewritten, so
`88d8102` is no longer an ancestor of `main`; the commit on `main` with tree `145748a5dce3` holds the same files
(`git log --format='%h %T' | grep 145748a5`). CI verifies the lock on every push. The class-asymmetry analysis and the
efficiency benchmark were added after scoring and are reported as post hoc.

## Training, serving and cost

### Training

<picture>
  <source media="(min-width: 960px) and (prefers-color-scheme: dark)" srcset="visual_outputs/12_training_tradeoffs_dark.png">
  <source media="(max-width: 959px) and (prefers-color-scheme: dark)" srcset="visual_outputs/12_training_tradeoffs_narrow_dark.png" width="360">
  <source media="(max-width: 959px)" srcset="visual_outputs/12_training_tradeoffs_narrow_light.png" width="360">
  <img src="visual_outputs/12_training_tradeoffs_light.png" width="830" alt="Three panels per controlled comparison: fit-time ratio, fit-memory ratio and paired official-test accuracy difference. Size, SVC: 15.5 times the fit time, 1.9 times the memory, +1.54 pp. Size, LinearSVC: 3.1, 2.7, +1.80 pp. LinearSVC versus logistic regression: 0.82, 0.97, −0.10 pp. SVC versus LinearSVC: 194, 2.7, −0.22 pp. 1–2-grams versus unigrams: 3.6, 3.9, +1.66 pp. Cleaning v2 versus legacy-v1: 1.3, 0.98, −0.01 pp. Final versus historical model: 0.38, 2.7, +3.42 pp.">
</picture>

Fit times are medians of three runs in fresh processes (one run for `svc_uni_full`), measured while other processes
kept 3–5 logical CPUs busy; ratios compare the two configurations of each pre-specified comparison. Training-set size
and word bigrams buy similar accuracy at different costs. Training on 25,000 instead of 8,000 reviews costs LinearSVC
3.1 times the fit time (1.07 to 3.30 s) for +1.80 pp [+1.51, +2.12], but costs the SVC 15.5 times (41.2 to 641 s) for
+1.54 pp [+1.20, +1.88]. 1–2-grams cost 3.6 times the fit time and 3.9 times the fit's memory increment (154 to 602
MB), as the vocabulary grows from 108,635 to 1,594,660 terms. Fitting the vectoriser takes 63–85% of every LinearSVC
and logistic-regression fit, and libsvm's optimisation at least 97% of every SVC fit. Complete tables: [benchmark
report](docs/BENCHMARKS.md#training).

### Serving

<picture>
  <source media="(min-width: 960px) and (prefers-color-scheme: dark)" srcset="visual_outputs/13_serving_latency_dark.png">
  <source media="(max-width: 959px) and (prefers-color-scheme: dark)" srcset="visual_outputs/13_serving_latency_narrow_dark.png" width="360">
  <source media="(max-width: 959px)" srcset="visual_outputs/13_serving_latency_narrow_light.png" width="360">
  <img src="visual_outputs/13_serving_latency_light.png" width="830" alt="Log-log chart of HTTP latency against sustained throughput over concurrency 1 to 32 for three served configurations, p50 solid and p95 dashed. Historical SVC: 98 requests/s at a 10.0 ms p50 at concurrency 1, saturating near 117 requests/s while p50 rises to 272 ms at concurrency 32. LinearSVC with unigrams: 420 requests/s at 2.3 ms, peaking at 688 requests/s. Final model: 341 requests/s at 2.8 ms, peaking at 544 requests/s. At concurrency 32 the p95 of both linear models rises to 180–230 ms.">
</picture>

<!-- table:serving -->
| Configuration | Model-only p50, ms | HTTP p50 / p95 at concurrency 1, ms | Peak throughput, requests/s (concurrency) | HTTP p50 / p95 at peak, ms | Server CPU s per 1,000 requests | Failed / completed |
|---|---:|---|---:|---|---:|---|
| `svc_uni_8k_historical` | 7.41 | 10.0 / 13.2 | 117 (8) | 68.0 / 78.1 | 8.70 | 0 / 19,638 |
| `linsvc_uni_full` | 0.59 | 2.3 / 2.9 | 688 (16) | 23.1 / 27.0 | 1.65 | 0 / 101,818 |
| `linsvc_bi_v2_full` | 0.90 | 2.8 / 4.0 | 544 (16) | 29.2 / 34.5 | 2.08 | 0 / 81,928 |
<!-- /table:serving -->

Medians over two sweeps, each with a fresh server; sweep ranges and every concurrency level are in the [benchmark
report](docs/BENCHMARKS.md#http-serving). The model-only p50 comes from the in-process benchmark under similar
background load. At concurrency 1, HTTP adds 1.7–2.6 ms to the p50 latency of single-review scoring. Throughput
saturates at concurrency 8–16, after which latency grows with the queue. At concurrency 32 the single-threaded client,
which shared the host, used up to 0.95 of a CPU, so that level is partly client-limited for the linear models. A fresh
server took 3.3–5.5 s from process start to its first successful `GET /model`.

### Inference economics

No paid API was called, so no charge was observed. Local costs below are estimates: measured CPU seconds per 1,000
predictions × price per vCPU-hour ÷ 3,600 ÷ utilisation, at a reference of 0.05 USD per vCPU-hour and 50%
utilisation, with the range over 0.02–0.10 USD and 25–100% utilisation. The prices are parameters, not quotes from a
provider, and cover CPU time only. Batch rows use in-process scoring at batch size 1,024; interactive rows use the
server's CPU time per completed HTTP request at its highest-throughput concurrency.

<!-- table:economics -->
| Configuration | Mode | CPU s per 1,000 predictions (measured) | USD per million predictions, reference (estimate) | Range over rates and utilisations (estimate) |
|---|---|---:|---:|---|
| `svc_uni_8k_historical` | batch, in-process | 5.30 | 0.15 | 0.029–0.59 |
| `svc_uni_8k_historical` | interactive HTTP | 8.70 | 0.24 | 0.048–0.97 |
| `svc_uni_full` | batch, in-process | 12.1 | 0.34 | 0.067–1.3 |
| `logreg_uni_full` | batch, in-process | 0.125 | 0.0035 | 0.00069–0.014 |
| `linsvc_uni_full` | batch, in-process | 0.133 | 0.0037 | 0.00074–0.015 |
| `linsvc_uni_full` | interactive HTTP | 1.65 | 0.046 | 0.0092–0.18 |
| `linsvc_uni_8k` | batch, in-process | 0.117 | 0.0033 | 0.00065–0.013 |
| `linsvc_bi_full` | batch, in-process | 0.320 | 0.0089 | 0.0018–0.036 |
| `linsvc_bi_v2_full` | batch, in-process | 0.492 | 0.014 | 0.0027–0.055 |
| `linsvc_bi_v2_full` | interactive HTTP | 2.08 | 0.058 | 0.012–0.23 |
<!-- /table:economics -->

At the reference point, the CPU time for a million predictions costs about 0.014 USD in batch and 0.058 USD over HTTP
for the final model, against 0.15 and 0.24 USD for the historical model. Prices and utilisation scale every
configuration alike, so the ratios hold at any price: the final model needs 0.093 times the historical model's batch
CPU and 0.24 times its server CPU per HTTP request. No LLM has been evaluated, so none appears in these comparisons.

## Behavioural robustness (provisional)

A suite of 237 constructed sentences in minimal pairs, label-preserving sets and single cases probes negation,
contrast, sarcasm and intensity ([`behavioural/`](behavioural/)). An AI assistant proposed every label and no person
has reviewed them, so these results are provisional and do not estimate error rates on natural reviews. On the
held-out template families, every trained configuration fails 81–89% of negation units and 75–88% of contrast units,
and none fails an intensity unit; sarcasm (10 units in 4 families) is too sparse to rank configurations. Unigram
configurations fail word-permutation pairs such as "The acting was great, but overall the film was dull." / "The
acting was dull, but overall the film was great." by construction, because both members have identical unigram TF-IDF
vectors. Table and intervals: [supplementary
results](docs/RESULTS.md#behavioural-suite-provisional-labels-not-adjudicated).

## Limitations and validity threats

- **Approximate grouping.** Movie identifiers are unavailable, so intervals resample blocks of 30 consecutive review
  ids as a proxy for movies. Movies that straddle block boundaries leave residual dependence, and intervals may be
  somewhat narrow; example-level and 60-id intervals differ from the 30-id intervals by at most 0.03 points.
- **One dataset.** All estimates describe English IMDB reviews with polar ratings; mixed reviews (ratings 5–6) are
  excluded by construction. They are not comparable with published results that use other training sets,
  preprocessing or protocols.
- **Untuned SVC.** The linear-kernel SVC keeps the historical C=1, so comparisons involving it confound the estimator
  with regularisation; its fit and inference costs also depend on C through the number of support vectors.
- **Post hoc efficiency analysis.** The benchmark and the Pareto set were produced after the test partition had
  been scored. The set uses point estimates, and neighbouring configurations can differ by less than their paired
  intervals. No configuration was selected, tuned or changed on these results.
- **One loaded machine.** All timings come from one Windows laptop CPU, single-threaded, while other processes kept
  3–12 of its 22 logical CPUs busy. Absolute times would be shorter on an idle machine, and servers, Linux, other
  CPUs, `httptools` or `uvloop`, or several workers would change them; the HTTP client shared the host with the
  server.
- **Benchmark workload.** Costs are measured on 2,000 training-partition reviews, not on the test reviews whose
  accuracy they are plotted against; per-review cost depends on review length.
- **Cost model.** Prices and utilisation are hypothetical; memory, storage, network and engineering costs are
  excluded.
- **Lexical overlap screen only.** The duplicate audit detects exact, normalised and near-lexical overlap between
  partitions; paraphrased duplicates escape it.
- **Unadjudicated behavioural labels.** The suite's labels are proposals by an AI assistant; its sentences are short
  and templated.
- **No LLM measurements.** The Claude baseline in [`study/llm/`](study/llm/) is implemented and validated offline but
  has not been run, so this repository reports no LLM accuracy, token count, charge or latency.

## Documentation

- [Reproduction guide](docs/REPRODUCING.md): environment, data download and checksums, and every command from
  training to figures.
- [Benchmark report](docs/BENCHMARKS.md): hardware, conditions, measurement boundaries, and the complete training,
  inference, HTTP and cost tables.
- [Supplementary results](docs/RESULTS.md): metrics with macro F1, paired comparisons with sensitivity intervals,
  train–test overlap, development selection, the behavioural suite and the historical evaluation.
- [Model card](docs/MODEL_CARD.md): configurations and artefact identifiers, intended use, failure modes, data
  attribution, the unexecuted LLM baseline and the serving contract.
- [Protocol](protocol/protocol_v1.json) and [lock](protocol/LOCK.json); [result files](results/README.md) with the
  command that produces each; [historical run report](visual_outputs/REPORT.md); [phone-width
  figures](docs/narrow-figures.md).

## Data and licence

Large Movie Review Dataset v1.0: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng and
Christopher Potts, "Learning Word Vectors for Sentiment Analysis", *Proceedings of ACL 2011*. The review text is not
redistributed here; the Hugging Face dataset card lists its licence as `other`. Code: [MIT](LICENSE).
