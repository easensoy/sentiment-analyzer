# Claude sentiment baseline runner

`study/llm/runner.py` evaluates Claude models as a zero-shot or few-shot baseline for binary
sentiment classification of IMDB reviews. It builds versioned prompts, requests a structured label,
records every API attempt in an append-only log, bounds spending before and during a run, and
summarises accuracy, token use, cost and timing per model and measurement mode. All API facts used
here (model identifiers, request and response shapes, headers, retry semantics, prompt caching,
the Message Batches API and prices) were taken from the claude-api skill, read on 2026-11-08.

## Files

| Path | Role |
| --- | --- |
| `study/llm/runner.py` | Library and command-line interface (`python -m study.llm.runner`). |
| `study/llm/prompts/v1/system.txt` | System prompt of prompt version `v1`. |
| `study/llm/prompts/v1/user.txt` | User-turn template; contains exactly one `{review}` slot. |
| `study/llm/prompts/v1/response_schema.json` | JSON Schema of the structured response. |
| `study/llm/pricing.json` | Per-model prices, batch multiplier and their provenance. |
| `tests/test_llm_runner.py` | Offline tests; every API exchange goes through `httpx.MockTransport`. |

## Commands

Run from the repository root. Without `--live`, `run` is a dry run: it builds and validates every
request body, prints the plan (eligible, completed, in-flight and pending examples; estimated input
tokens, maximum output tokens and the cost bound per model) and neither opens a network connection
nor writes a file.

```bash
# Dry run: zero-shot, default models, online mode
python -m study.llm.runner run --examples EVAL.csv

# Live online run with few-shot exemplars and a spending limit for this invocation
python -m study.llm.runner run --examples EVAL.csv --exemplars SHOTS.csv --live --max-spend-usd 25

# Live batch run (Message Batches API) for two models
python -m study.llm.runner run --examples EVAL.csv --mode batch --models claude-haiku-5-5,claude-sonnet-5-5 \
    --live --max-spend-usd 10

# Resume: repeat the same command; completed examples are skipped
# Summarise one or more record files
python -m study.llm.runner summarize --records results/llm/RUN_ID/requests.jsonl --examples EVAL.csv
```

`run` options: `--models` (comma-separated; default `claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5`,
cheapest first), `--mode online|batch`, `--prompt-version` (default `v1`), `--max-review-chars`,
`--max-tokens` (default 1024), `--cache-ttl 5m|1h|off` (default `5m`), `--max-attempts` (default 5),
`--concurrency` (online mode, 1 to 16, default 1), `--run-id`, `--results-dir` (default `results/llm`),
`--records`, `--live`, `--max-spend-usd`, `--timeout-s` (default 120), `--poll-interval-s`
(default 60) and `--batch-max-wait-s` (default 86,400). Live runs read the API key from
`ANTHROPIC_API_KEY` when a request is sent; the key is sent only in the `x-api-key` header and is
never written to any output.

## Inputs and the eligible set

`--examples` and `--exemplars` are CSV files (optionally gzip-compressed, `.gz`) with the columns
`example_id`, `text` and `label`, where label 0 is negative and 1 is positive. Further columns are
ignored. Identifiers must be unique and texts non-empty.

The eligible set is every row of `--examples`; each model in a run is evaluated on that same set.
Its identity is `eval_set_sha256`: the sha256 of the lines `example_id<TAB>sha256(text)`, sorted by
`example_id`. Exemplars must satisfy three conditions, otherwise the run stops before any request:
every exemplar `example_id` starts with `train/` (development data from the official training
partition); no exemplar `example_id` occurs among the evaluation examples; and no exemplar text
occurs among the evaluation texts, compared after collapsing whitespace and casefolding. The
exemplar set is identified by the sha256 of `example_id<TAB>sha256(text)<TAB>label` lines in prompt
order. Example identifiers are never placed in a prompt.

## Prompts and request construction

A prompt version is a directory `study/llm/prompts/<version>/` containing `system.txt`, `user.txt`
and `response_schema.json`. File hashes are computed over the file text after newline normalisation
to LF, so a CRLF checkout yields the same hashes. `prompt_sha256`, recorded with every request, is
the sha256 of the canonical JSON mapping of the three file names to their hashes.

Every request is a Messages API body (`anthropic-version: 2026-10-21`) with:

- `system`: one text block with the system prompt.
- `messages`: for each exemplar, a user turn (the user template applied to the exemplar text) and an
  assistant turn containing its answer, `{"label": "positive"}` or `{"label": "negative"}`; then the
  user turn for the review under evaluation. The conversation ends with a user turn, because
  assistant prefill is rejected by the current models.
- `cache_control` on the last block of the shared prefix: the last exemplar answer, or the system
  block in a zero-shot run. The prefix is byte-identical across requests, so prompt caching applies;
  the varying review follows the breakpoint. `--cache-ttl 1h` selects the one-hour TTL and `off`
  removes the breakpoint. A prefix shorter than the model's minimum cacheable length (512 tokens on
  the default models, per the skill) is not cached and reports zero cache tokens; the zero-shot `v1`
  system prompt alone (about 650 characters) is expected to fall below that length, so cache usage
  arises mainly in few-shot runs. A cache entry becomes readable only after the first response that
  writes it has started, so concurrent requests may each write the entry, and cache hits inside a
  batch are best-effort.
- `output_config.format`: `{"type": "json_schema", "schema": <response_schema.json>}`, the
  structured-output mechanism the skill recommends for raw HTTP requests. The schema admits exactly
  `{"label": "negative"}` and `{"label": "positive"}`.
- Model-specific fields from `MODEL_REQUEST_OPTIONS` in `runner.py`, the only place where the
  runner distinguishes models. All four configured models receive `output_config.effort: "low"`,
  the skill's suggested starting point for classification. No `thinking` field is sent, so every
  model runs its default adaptive thinking (Claude Opus 5.5 and Claude Sonnet 5.5 reject disabled
  thinking); thinking tokens count toward `max_tokens` and are billed as output. No `temperature`,
  `top_p` or `top_k` is sent, because API version 2026-10-21 rejects them on every model; outputs
  are therefore sampled with default settings.

The server-side refusal fallback that the skill recommends by default is deliberately not enabled:
it would replace a refused answer with another model's answer and so confound the per-model
measurement, and the Message Batches API rejects the parameter. Refusals are recorded as such.

The dry run validates every body: required fields, no sampling parameters, alternating roles that
start and end with a user turn, non-empty text blocks, at most four cache breakpoints, a
`json_schema` output format and, in batch mode, a valid `custom_id`.

## Truncation policy

Truncation is off by default, because the longest IMDB review has about 13,700 characters, far
within the context window of every configured model. With `--max-review-chars N` (N at least 100),
a review longer than N characters (Unicode code points) is replaced by its first and last
characters joined by the fixed marker `\n[... middle of review omitted ...]\n`; the result has at
most N characters, and the head receives one more character than the tail when the remainder is
odd. The rule is deterministic, applies to exemplars as well, and is part of the run configuration.
Each record states `truncated` and `original_chars`.

## Response classification

A successful response is classified by `stop_reason` before its content is read, and text blocks are
selected by type (thinking blocks precede them). Each attempt receives one `parse_status`.

| `parse_status` | Terminal | Meaning |
| --- | --- | --- |
| `ok` | yes | `end_turn`, and the text parses as JSON that conforms to the schema. |
| `malformed` | yes | Unparseable or schema-violating content, or an unexpected stop reason. |
| `refusal` | yes | `stop_reason` is `refusal` (`stop_details` is recorded verbatim). |
| `max_tokens` | yes | Output truncated by `max_tokens`. |
| `error` | yes | Non-retryable HTTP error (400, 413 and other 4xx outside the abort set). |
| `retryable_error` | no | Retryable failure on an attempt that was not the last permitted one. |
| `missing` | no | No completed result after all attempts, or no usable result from a batch. |
| `not_attempted` | no | Not sent because the spending limit was reached. |
| `aborted` | no | Run-level failure (401, 402, 403 or 404); the run stops. |

`raw_text` holds the model's text output, or the error body for failed requests, truncated to 2,000
characters.

## Retries and run-level errors

HTTP 429, 500, 502, 503, 504 and 529 (overloaded) and transport failures (timeouts, connection and
protocol errors) are retried up to `--max-attempts` attempts per example per invocation. The wait
before attempt k+1 is the `retry-after` header value when present (delta-seconds or HTTP date),
otherwise `min(2^(k-1) + U(0, 1), 60)` seconds. Every attempt is recorded. HTTP 401, 402, 403 and
404 concern credentials, billing, permissions or the model identifier rather than the example, so
the run stops and the example stays open. Other client errors are terminal `error` records; a
request defect that affects every example therefore closes every example, which is why a first live
run on a small examples file is advisable. Batch status and result downloads use the same retry
policy; a batch creation is resent only after HTTP 429 or 529 or when the request never left the
client, because any other failure may have created the batch.

## Records and resumption

A run lives in `results/llm/<run_id>/` (`--records` relocates `requests.jsonl`; the other files are
written beside it):

- `requests.jsonl`: one record per attempt, appended, flushed and fsynced before the next action.
- `run.json`: the manifest, holding the configuration, its hash, per-model request options and one
  entry per invocation.
- `batches.jsonl`: batch-mode submission and collection events.

The default `run_id` is `<prompt version>-<first 12 hex digits of the configuration hash>`. The
configuration covers the API version, prompt version and file hashes, eligible-set and exemplar-set
hashes, exemplar order, truncation limit and marker, `max_tokens` and cache TTL. Repeating a command
therefore resumes the same run, and a run directory refuses an invocation whose configuration or
per-model request options differ from its manifest. Models may be added to an existing run.

Record fields: `run_id`, `mode`, `model`, `prompt_version`, `prompt_sha256`, `example_id`, `attempt`
(continuing across invocations; null for `not_attempted`), `request_started_utc`, `latency_s`
(online only), `http_status`, `request_id` (the `request-id` response header), `stop_reason`,
`stop_details`, `usage` (verbatim), `cost_usd`, `batch_discount_applied`, `parsed_label` (0 or 1,
only when `ok`), `parse_status`, `truncated`, `original_chars`, `raw_text`, `error_type`, and in
batch mode `batch_id`, `batch_created_utc`, `batch_ended_utc`, `batch_completion_s` and
`batch_result_type`.

On restart the runner skips every example that has a terminal record for the same model, prompt
version and mode in the run, and sends the rest. Online and batch measurements are kept apart: a
batch invocation does not skip examples completed online. A line left incomplete by an interrupted
write is skipped when reading and terminated before the next append, so an example whose final
record was cut short is requested again. In batch mode, batches that were submitted but not
collected are polled and collected before any new submission; their examples are never resubmitted
while the batch is pending.

## Spending control

Live runs require both `--live` and `--max-spend-usd X`; X limits the spend of one invocation.
Before any request, the runner computes an upper bound for every pending request: input tokens are
estimated as `ceil(characters / 3)` over the system prompt, all message texts (exemplars and
review) and the serialised response schema, and priced at the higher of the model's input and
cache-write rates (no cache-read discount); output is priced at `max_tokens`; the batch multiplier
applies in batch mode. If the sum exceeds X, the run is refused and nothing is sent.

During the run, each request reserves its bound before it is sent and the reservation is replaced
by the realized cost when the response arrives. Once realized cost plus outstanding reservations
plus the next bound would exceed X, no further request is sent, and every example of the invocation
that was never sent is recorded as `not_attempted`, so a later invocation resumes it. In batch mode
the reservation covers each request of a batch before submission. The realized cost can exceed X
only if a response costs more than its estimated bound, which the heuristic does not exclude.
Batches recovered from an earlier invocation are collected without being charged to the current
limit.

## Measurement modes

Online and batch results are reported separately.

- **Online**: Messages API requests, sequential by default, or with a fixed `--concurrency`. The
  latency of a request is the wall time of one non-streaming HTTP request, including model thinking
  and generation and network transfer; quantiles use only attempts that returned HTTP 200.
  Concurrency changes queueing and therefore latency, so it should be reported with the results.
- **Batch**: one Message Batches submission per model, split at 100,000 requests or 200 MB of request
  body (below the documented 256 MB limit). Each request's `custom_id` is the unpadded URL-safe
  base64 encoding of its `example_id`; results are mapped back by `custom_id`, never by position.
  Batches are polled every `--poll-interval-s` seconds until they end or `--batch-max-wait-s` passes,
  in which case a later invocation collects them. The completion time of a batch is `ended_at` minus
  `created_at` as reported by the API (local submission and observation times when the API omits
  them); it is not a per-request latency. A `succeeded` item is classified like an online response;
  an `errored` item of type `invalid_request_error` or `request_too_large` is `error`; other errored
  items, `canceled`, `expired` and requests absent from the results are `missing`.

## Summary

`summarize` reads one or more record files and the examples file, checks that the examples file is
the eligible set of each run whose manifest is present, and writes `summary.json` and
`per_example.csv` (default directory: that of the first records file). It reports one group per
(run, model, mode, prompt version) that has records; models and modes without records produce no
output. Definitions, also written into `summary.json`:

- Final status of an example: its first terminal record; otherwise `not_attempted` if its last
  record is a spending-limit marker, else `missing` (including examples with no record).
- `accuracy_all_eligible`: correct / eligible. Every example whose final status is not `ok` counts
  as incorrect.
- `accuracy_parsed_ok`: correct / examples with final status `ok`, with that denominator.
- `error_rate_by_true_class`: for each true class, misclassified / examples of that class, over all
  eligible examples (non-`ok` counts as an error) and over the parsed-`ok` subset.
- `tokens`: input, output, cache-write and cache-read tokens summed over all attempts, and the mean
  per example with at least one response carrying usage.
- `cost_usd`: the sum of recorded costs, and its per-example mean on the same denominator.
- `latency_s` (online): number of requests, median, p90 and p99 (linear interpolation).
- `batches` (batch): per-batch creation, end and completion times, and the span from the first
  creation to the last end.

Every ratio is reported with its numerator and denominator; a ratio with a zero denominator is null.
`per_example.csv` has one row per eligible example and group.

## Prices

`pricing.json` records its source as "claude-api skill, read on 2026-11-08". Rates are USD per
million tokens.

| Model | Input | Output | Cache write 5 min | Cache write 1 h | Cache read |
| --- | ---: | ---: | ---: | ---: | ---: |
| `claude-haiku-5-5` (prompt up to 100K tokens) | 0.10 | 0.50 | 0.125 | 0.20 | 0.01 |
| `claude-haiku-5-5` (prompt above 100K tokens) | 0.50 | 2.50 | 0.625 | 1.00 | 0.05 |
| `claude-sonnet-5-5` | 2.00 | 10.00 | 2.50 | 4.00 | 0.20 |
| `claude-opus-5-5` | 4.00 | 20.00 | 5.00 | 8.00 | 0.20 |
| `claude-fable-5-1` (not a default) | 10.00 | 50.00 | 12.50 | 20.00 | 0.25 |

The Claude Haiku 5.5 cache rates are derived from the multipliers the skill states (cache reads
0.1x, five-minute writes 1.25x, one-hour writes 2x the input rate); the other cache rates are stated
directly. The Message Batches API charges 50% of these rates on every token class, including cache
reads and writes. The cost of a response is computed from its reported usage: uncached input,
cache writes by TTL (from `usage.cache_creation` when present), cache reads and output tokens, the
latter including thinking. For refusals that occur before any output, billing depends on the
refusal category, so the recorded cost may overstate the amount charged. Prices should be checked
against the current Anthropic pricing page before a budget is fixed.

## Limitations

- **Contamination.** IMDB reviews are public and widely redistributed, so contamination of the
  evaluated models' training data is plausible and unquantified; results may overstate
  generalization.
- **Sampling.** Sampling parameters cannot be set on API version 2026-10-21 and adaptive thinking
  remains on, so responses are stochastic. Each example is classified once; run-to-run variance is
  not estimated.
- **Prompt and effort dependence.** Results hold for prompt `v1`, the exemplar set and effort `low`;
  other prompts, exemplars or effort levels can change accuracy, token use and latency.
- **Model identity.** The model identifiers carry no snapshot date; the manifest records when each
  invocation ran.
- **Cost estimate.** The three-characters-per-token estimate is a heuristic. It does not count the
  instructions the API adds for structured outputs, and tokenizers differ between models.
- **API details outside the skill.** The skill documents batch creation (`POST /v1/messages/batches`),
  status polling, result types and the SDK result shape. The REST paths used for status
  (`GET /v1/messages/batches/{id}`) and results (`GET /v1/messages/batches/{id}/results`, one JSON
  object per line), the nested shape of errored results and the `custom_id` constraint
  (1 to 64 characters from `[A-Za-z0-9_-]`) follow the public Message Batches API reference and were
  not re-verified online when this runner was written; a first batch run on a few examples confirms
  them.
