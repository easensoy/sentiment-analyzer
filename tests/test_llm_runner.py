"""Offline tests for study.llm.runner: every API exchange goes through httpx.MockTransport."""
import base64
import csv
import io
import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from study.llm import runner

DUMMY_KEY = "sk-ant-test-0000-dummy-not-a-real-key"
HAIKU = "claude-haiku-5-5"
EVAL = [("test/neg/1", 0), ("test/neg/2", 0), ("test/pos/3", 1), ("test/pos/4", 1)]
SHOTS = [("train/pos/10", 1), ("train/neg/11", 0)]
USAGE = {"input_tokens": 400, "output_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
         "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
         "service_tier": "standard"}
HAIKU_COST = (400 * 0.10 + 40 * 0.50) / 1_000_000  # pricing.json: $0.10 input, $0.50 output per MTok


@pytest.fixture(autouse=True)
def dummy_api_key(monkeypatch):
    """Every test sees a dummy key, so the runner never reads a real one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def eval_csv(tmp_path):
    return write_examples(tmp_path / "eval.csv", EVAL)


class FakeClock:
    """Deterministic time: monotonic() advances 0.25 s per call and sleep() by its argument."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        self.now += 0.25
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def utc_now(self):
        return datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=self.now - 1000.0)

    def clock(self):
        return runner.Clock(sleep=self.sleep, monotonic=self.monotonic, utc_now=self.utc_now)


def review_text(example_id):
    return f"Review {example_id}: an ordinary film with a plot."


def write_examples(path, rows, texts=None):
    texts = texts or {}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["example_id", "text", "label"])
        for example_id, label in rows:
            writer.writerow([example_id, texts.get(example_id, review_text(example_id)), label])
    return path


def message(label="positive", *, stop_reason="end_turn", text=None, usage=None, stop_details=None):
    """A Messages API response body; a (signed, empty) thinking block precedes the text."""
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    output = json.dumps({"label": label}) if text is None and label is not None else text
    if output is not None:
        content.append({"type": "text", "text": output})
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": HAIKU, "content": content,
            "stop_reason": stop_reason, "stop_details": stop_details, "usage": usage or USAGE}


def ok(label):
    return httpx.Response(200, json=message(label), headers={"request-id": "req_ok"})


def api_error(status, error_type, headers=None, text="error"):
    return httpx.Response(status, headers=headers,
                          json={"type": "error", "error": {"type": error_type, "message": text}, "request_id": "req_err"})


class FakeMessagesAPI:
    """MockTransport handler for POST /v1/messages. Scripted steps per example id (a response, or an
    exception to raise); once an example's script is used up, it answers with the true label."""

    def __init__(self, script=None):
        self.script = {example_id: list(steps) for example_id, steps in (script or {}).items()}
        self.calls = []

    def __call__(self, request):
        assert (request.method, request.url.path) == ("POST", "/v1/messages")
        body = json.loads(request.content)
        example_id = re.search(r"Review (\S+):", body["messages"][-1]["content"][0]["text"]).group(1)
        self.calls.append((example_id, request, body))
        steps = self.script.get(example_id)
        step = steps.pop(0) if steps else ok("positive" if example_id.startswith("test/pos") else "negative")
        if isinstance(step, Exception):
            raise step
        return step


class FakeBatchAPI:
    """MockTransport handler for the Message Batches endpoints: one batch, which ends on the
    `polls_until_end`-th status request; results come from `outcomes` (example id -> result)."""

    def __init__(self, outcomes, polls_until_end=2, crash_on_poll=False):
        self.outcomes, self.polls_until_end, self.crash_on_poll = outcomes, polls_until_end, crash_on_poll
        self.calls, self.created, self.polls = [], [], 0

    def batch(self, ended):
        return {"id": "msgbatch_test", "type": "message_batch",
                "processing_status": "ended" if ended else "in_progress",
                "created_at": "2026-11-08T12:00:00Z", "ended_at": "2026-11-08T12:30:00Z" if ended else None,
                "request_counts": {"processing": 0 if ended else 4, "succeeded": 0, "errored": 0, "canceled": 0,
                                   "expired": 0}}

    def __call__(self, request):
        path = request.url.path
        self.calls.append((request.method, path))
        assert request.headers["x-api-key"] == DUMMY_KEY
        if (request.method, path) == ("POST", "/v1/messages/batches"):
            self.created.append(json.loads(request.content))
            return httpx.Response(200, json=self.batch(ended=False))
        if path == "/v1/messages/batches/msgbatch_test":
            if self.crash_on_poll:
                raise RuntimeError("simulated crash while polling")
            self.polls += 1
            return httpx.Response(200, json=self.batch(ended=self.polls >= self.polls_until_end))
        assert path == "/v1/messages/batches/msgbatch_test/results"
        lines = [json.dumps({"custom_id": runner.encode_custom_id(example_id), "result": result})
                 for example_id, result in self.outcomes.items()]
        return httpx.Response(200, content="\n".join(lines).encode("utf-8"))


def settings(tmp_path, **overrides):
    values = dict(examples_path=tmp_path / "eval.csv", models=(HAIKU,), results_dir=tmp_path / "results",
                  run_id="test-run", live=True, max_spend_usd=5.0)
    values.update(overrides)
    return runner.RunSettings(**values)


def execute(tmp_path, handler, clock, **overrides):
    return runner.run(settings(tmp_path, **overrides), transport=httpx.MockTransport(handler), clock=clock.clock(),
                      rng=random.Random(0), out=io.StringIO())


def records_of(report):
    return runner.read_jsonl(report.records_path)[0]


def attempts_for(report, example_id):
    return [r for r in records_of(report) if r["example_id"] == example_id]


def terminal_counts(report):
    return Counter(r["example_id"] for r in records_of(report) if r["parse_status"] in runner.TERMINAL_STATUSES)


# --- Responses ----------------------------------------------------------------------------------
def test_valid_structured_response_is_parsed_costed_and_recorded(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI()
    report = execute(tmp_path, api, clock)
    records = records_of(report)
    assert len(api.calls) == 4 and len(records) == 4
    assert {r["example_id"]: (r["parse_status"], r["parsed_label"]) for r in records} == \
        {example_id: ("ok", label) for example_id, label in EVAL}
    first = records[0]
    assert (first["attempt"], first["http_status"], first["request_id"], first["mode"]) == (1, 200, "req_ok", "online")
    assert first["latency_s"] == pytest.approx(0.25)
    assert first["usage"] == USAGE and first["raw_text"] == '{"label": "negative"}'
    assert first["cost_usd"] == pytest.approx(HAIKU_COST) and first["batch_discount_applied"] is False
    assert first["prompt_version"] == "v1" and len(first["prompt_sha256"]) == 64
    _, request, body = api.calls[0]
    assert request.headers["anthropic-version"] == "2026-10-21"
    assert request.headers["x-api-key"] == DUMMY_KEY
    schema = json.loads((runner.PROMPTS_DIR / "v1" / "response_schema.json").read_text(encoding="utf-8"))
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": schema}, "effort": "low"}
    assert not set(runner.SAMPLING_PARAMETERS) & set(body) and "thinking" not in body
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("response, status", [
    (message(None, stop_reason="refusal", stop_details={"type": "refusal", "category": "bio", "explanation": None}),
     "refusal"),
    (message(None, stop_reason="max_tokens"), "max_tokens"),
    (message(text="The review is positive."), "malformed"),
    (message(text='{"label": "neutral"}'), "malformed"),
    (message(text='{"label": "positive", "confidence": 0.9}'), "malformed"),
    (message("positive", stop_reason="tool_use"), "malformed"),
], ids=["refusal", "max_tokens", "unparseable", "enum_violation", "extra_key", "unexpected_stop"])
def test_terminal_response_classes(tmp_path, eval_csv, clock, response, status):
    api = FakeMessagesAPI({"test/pos/3": [httpx.Response(200, json=response)]})
    report = execute(tmp_path, api, clock)
    (record,) = attempts_for(report, "test/pos/3")
    assert (record["parse_status"], record["parsed_label"], record["attempt"]) == (status, None, 1)
    assert record["stop_reason"] == response["stop_reason"] and record["stop_details"] == response["stop_details"]
    assert len(api.calls) == 4 and clock.sleeps == []


def test_rate_limit_then_success_honours_retry_after(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/neg/1": [api_error(429, "rate_limit_error", headers={"retry-after": "7"})]})
    records = attempts_for(execute(tmp_path, api, clock), "test/neg/1")
    assert [(r["attempt"], r["http_status"], r["parse_status"]) for r in records] == \
        [(1, 429, "retryable_error"), (2, 200, "ok")]
    assert records[0]["error_type"] == "rate_limit_error" and records[0]["request_id"] == "req_err"
    assert clock.sleeps == [7.0]
    assert len(api.calls) == 5


def test_overloaded_529_then_success_backs_off_with_jitter(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/neg/1": [api_error(529, "overloaded_error")]})
    records = attempts_for(execute(tmp_path, api, clock), "test/neg/1")
    assert [(r["http_status"], r["parse_status"]) for r in records] == [(529, "retryable_error"), (200, "ok")]
    assert records[0]["error_type"] == "overloaded_error"
    assert len(clock.sleeps) == 1 and 1.0 <= clock.sleeps[0] <= 2.0  # base 1 s plus up to 1 s jitter


def test_network_timeout_is_retried(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/neg/1": [httpx.ReadTimeout("read timed out")]})
    records = attempts_for(execute(tmp_path, api, clock), "test/neg/1")
    assert [(r["http_status"], r["parse_status"], r["error_type"]) for r in records] == \
        [(None, "retryable_error", "ReadTimeout"), (200, "ok", None)]
    assert len(clock.sleeps) == 1


def test_non_retryable_400_is_a_terminal_error(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/neg/1": [api_error(400, "invalid_request_error")]})
    (record,) = attempts_for(execute(tmp_path, api, clock), "test/neg/1")
    assert (record["http_status"], record["parse_status"], record["error_type"]) == (400, "error", "invalid_request_error")
    assert clock.sleeps == []
    again = FakeMessagesAPI()
    execute(tmp_path, again, clock)
    assert again.calls == []  # every example already has a terminal record


def test_exhausted_retries_leave_the_example_missing_and_resumable(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/neg/1": [api_error(500, "api_error") for _ in range(3)]})
    report = execute(tmp_path, api, clock, max_attempts=3)
    records = attempts_for(report, "test/neg/1")
    assert [(r["attempt"], r["parse_status"]) for r in records] == \
        [(1, "retryable_error"), (2, "retryable_error"), (3, "missing")]
    assert len(clock.sleeps) == 2
    again = FakeMessagesAPI()
    report = execute(tmp_path, again, clock, max_attempts=3)
    assert [call[0] for call in again.calls] == ["test/neg/1"]
    assert attempts_for(report, "test/neg/1")[-1]["attempt"] == 4


def test_authentication_error_aborts_without_closing_the_example(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/neg/1": [api_error(401, "authentication_error")]})
    with pytest.raises(runner.RunAborted, match="authentication_error"):
        execute(tmp_path, api, clock)
    assert len(api.calls) == 1
    report = execute(tmp_path, FakeMessagesAPI(), clock)
    assert [r["parse_status"] for r in attempts_for(report, "test/neg/1")] == ["aborted", "ok"]


# --- Resumability -------------------------------------------------------------------------------
def test_resume_requests_only_examples_without_a_terminal_record(tmp_path, eval_csv, clock):
    failing = {example_id: [api_error(503, "api_error")] for example_id in ("test/neg/2", "test/pos/4")}
    first = FakeMessagesAPI(failing)
    execute(tmp_path, first, clock, max_attempts=1)
    assert len(first.calls) == 4
    second = FakeMessagesAPI()
    report = execute(tmp_path, second, clock, max_attempts=1)
    assert sorted(call[0] for call in second.calls) == ["test/neg/2", "test/pos/4"]
    assert terminal_counts(report) == Counter({example_id: 1 for example_id, _ in EVAL})


def test_crash_mid_run_then_resume_yields_one_terminal_record_per_example(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/pos/3": [RuntimeError("simulated crash")]})
    with pytest.raises(RuntimeError, match="simulated crash"):
        execute(tmp_path, api, clock)
    records_path = tmp_path / "results" / "test-run" / "requests.jsonl"
    assert [r["example_id"] for r in runner.read_jsonl(records_path)[0]] == ["test/neg/1", "test/neg/2"]
    with records_path.open("ab") as handle:  # a record cut short by the crash
        handle.write(b'{"run_id":"test-run","mode":"onl')
    again = FakeMessagesAPI()
    report = execute(tmp_path, again, clock)
    assert [call[0] for call in again.calls] == ["test/pos/3", "test/pos/4"]
    records, unreadable = runner.read_jsonl(records_path)
    assert unreadable == 1 and len(records) == 4
    assert terminal_counts(report) == Counter({example_id: 1 for example_id, _ in EVAL})


def test_concurrent_online_requests_record_each_example_once(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI()
    report = execute(tmp_path, api, clock, concurrency=3)
    assert len(api.calls) == 4 and terminal_counts(report) == Counter({example_id: 1 for example_id, _ in EVAL})


def test_a_changed_configuration_cannot_reuse_a_run(tmp_path, eval_csv, clock):
    execute(tmp_path, FakeMessagesAPI(), clock)
    with pytest.raises(runner.InputError, match="max_tokens"):
        execute(tmp_path, FakeMessagesAPI(), clock, max_tokens=2048)


# --- Costs and spending -------------------------------------------------------------------------
def test_cost_from_usage_uses_pricing_json_including_cache_tokens():
    pricing = runner.load_pricing()
    assert pricing.source == "claude-api skill, read on 2026-11-08" and pricing.batch_multiplier == 0.5
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 3000,
             "cache_read_input_tokens": 5000,
             "cache_creation": {"ephemeral_5m_input_tokens": 3000, "ephemeral_1h_input_tokens": 0}}
    # Claude Sonnet 5.5 per MTok: input $2, 5-minute cache write $2.50, cache read $0.20, output $10.
    sonnet = (1000 * 2.00 + 3000 * 2.50 + 5000 * 0.20 + 200 * 10.00) / 1_000_000
    assert pricing.cost_usd("claude-sonnet-5-5", usage, batch=False, cache_ttl="5m") == pytest.approx(sonnet)
    assert pricing.cost_usd("claude-sonnet-5-5", usage, batch=True, cache_ttl="5m") == pytest.approx(sonnet / 2)
    # Claude Opus 5.5 per MTok: input $4, 1-hour cache write $8, cache read $0.20, output $20.
    hour = {**usage, "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 3000}}
    opus = (1000 * 4.00 + 3000 * 8.00 + 5000 * 0.20 + 200 * 20.00) / 1_000_000
    assert pricing.cost_usd("claude-opus-5-5", hour, batch=False, cache_ttl="5m") == pytest.approx(opus)
    # Writes without a TTL breakdown are priced at the run's TTL.
    unsplit = {key: value for key, value in usage.items() if key != "cache_creation"}
    assert pricing.cost_usd("claude-opus-5-5", unsplit, batch=False, cache_ttl="1h") == pytest.approx(opus)
    # Claude Haiku 5.5 switches to its $0.50 / $2.50 card for prompts above 100K tokens.
    assert pricing.cost_usd(HAIKU, {"input_tokens": 100_000, "output_tokens": 100}, batch=False, cache_ttl="5m") == \
        pytest.approx((100_000 * 0.10 + 100 * 0.50) / 1_000_000)
    assert pricing.cost_usd(HAIKU, {"input_tokens": 150_000, "output_tokens": 100}, batch=False, cache_ttl="5m") == \
        pytest.approx((150_000 * 0.50 + 100 * 2.50) / 1_000_000)


def test_preflight_refuses_when_the_bound_exceeds_the_limit(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI()
    with pytest.raises(runner.PreflightRefused, match="exceeds --max-spend-usd"):
        execute(tmp_path, api, clock, max_spend_usd=1e-6)
    assert api.calls == [] and not (tmp_path / "results").exists()


def test_runtime_stop_before_the_limit_marks_the_rest_not_attempted(tmp_path, eval_csv, clock):
    limit = runner.run(settings(tmp_path, live=False), out=io.StringIO()).cost_bound_usd  # pre-flight just passes
    heavy = {**USAGE, "input_tokens": int(0.9 * limit / 0.10 * 1_000_000)}  # far above the estimate
    api = FakeMessagesAPI({"test/neg/1": [httpx.Response(200, json=message("negative", usage=heavy))]})
    report = execute(tmp_path, api, clock, max_spend_usd=limit)
    assert len(api.calls) == 1 and report.spend_limit_reached
    assert report.realized_cost_usd <= limit
    assert {r["example_id"]: r["parse_status"] for r in records_of(report)} == {
        "test/neg/1": "ok", "test/neg/2": "not_attempted", "test/pos/3": "not_attempted",
        "test/pos/4": "not_attempted"}
    again = FakeMessagesAPI()
    execute(tmp_path, again, clock)
    assert sorted(call[0] for call in again.calls) == ["test/neg/2", "test/pos/3", "test/pos/4"]


def test_dry_run_makes_no_transport_calls_and_writes_nothing(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI()
    out = io.StringIO()
    shots = write_examples(tmp_path / "shots.csv", SHOTS)
    report = runner.run(settings(tmp_path, live=False, max_spend_usd=None, exemplars_path=shots,
                                 models=runner.DEFAULT_MODELS),
                        transport=httpx.MockTransport(api), clock=clock.clock(), out=out)
    assert api.calls == [] and not (tmp_path / "results").exists()
    assert report.live is False and list(report.plans) == list(runner.DEFAULT_MODELS)
    assert all(len(plan.pending) == 4 and plan.cost_bound_usd > 0 for plan in report.plans.values())
    assert "Dry run" in out.getvalue() and "claude-opus-5-5" in out.getvalue()


def test_cli_requires_a_spend_limit_for_live_runs(tmp_path, eval_csv, capsys):
    with pytest.raises(SystemExit) as stopped:
        runner.main(["run", "--examples", str(eval_csv), "--live"])
    assert stopped.value.code == 2 and "--max-spend-usd" in capsys.readouterr().err
    assert runner.main(["run", "--examples", str(eval_csv), "--results-dir", str(tmp_path / "results")]) == 0
    assert not (tmp_path / "results").exists()


# --- Prompts, exemplars and truncation ----------------------------------------------------------
def test_few_shot_exemplars_are_prior_turns_inside_the_cached_prefix(tmp_path, eval_csv, clock):
    shots = write_examples(tmp_path / "shots.csv", SHOTS)
    api = FakeMessagesAPI()
    execute(tmp_path, api, clock, exemplars_path=shots)
    body = api.calls[0][2]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert [m["content"][0]["text"] for m in body["messages"][1:4:2]] == ['{"label": "positive"}', '{"label": "negative"}']
    assert "Review train/pos/10:" in body["messages"][0]["content"][0]["text"]
    assert body["messages"][3]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["system"][0] and "cache_control" not in body["messages"][-1]["content"][0]
    prefixes = {json.dumps([call[2]["system"], call[2]["messages"][:-1]]) for call in api.calls}
    assert len(prefixes) == 1  # byte-identical shared prefix, so the cache can be read
    manifest = json.loads((tmp_path / "results" / "test-run" / "run.json").read_text(encoding="utf-8"))
    assert manifest["config"]["exemplar_ids"] == ["train/pos/10", "train/neg/11"]
    assert manifest["config"]["prompt_files_sha256"].keys() == {"system.txt", "user.txt", "response_schema.json"}


@pytest.mark.parametrize("shot_rows, eval_rows, texts, problem", [
    ([("test/pos/9", 1)], EVAL, {}, "outside the training partition"),
    ([("train/pos/10", 1)], EVAL + [("train/pos/10", 1)], {}, "ids also among the evaluation examples"),
    ([("train/pos/77", 1)], EVAL, {"train/pos/77": "  REVIEW test/pos/3: an ordinary   film with a plot. "},
     "texts also among the evaluation examples"),
], ids=["not_train", "shared_id", "shared_text"])
def test_exemplar_validation(tmp_path, clock, shot_rows, eval_rows, texts, problem):
    write_examples(tmp_path / "eval.csv", eval_rows)
    shots = write_examples(tmp_path / "shots.csv", shot_rows, texts)
    api = FakeMessagesAPI()
    with pytest.raises(runner.InputError, match=problem):
        execute(tmp_path, api, clock, exemplars_path=shots)
    assert api.calls == []


def test_truncation_keeps_head_and_tail_and_is_recorded(tmp_path, clock):
    text = "A" * 300 + "B" * 300
    truncated, flag = runner.truncate_review(text, 200)
    assert flag and len(truncated) == 200 and runner.TRUNCATION_MARKER in truncated
    assert truncated.startswith("A") and truncated.endswith("B")
    assert runner.truncate_review(text, 200) == (truncated, True)
    assert runner.truncate_review(text, 600) == (text, False) and runner.truncate_review(text, None) == (text, False)
    long_review = review_text("test/pos/3") + " middle" * 100 + " the ending"
    write_examples(tmp_path / "eval.csv", EVAL, texts={"test/pos/3": long_review})
    api = FakeMessagesAPI()
    report = execute(tmp_path, api, clock, max_review_chars=150)
    flags = {r["example_id"]: (r["truncated"], r["original_chars"]) for r in records_of(report)}
    assert flags["test/pos/3"] == (True, len(long_review)) and flags["test/neg/1"] == (False, len(review_text("test/neg/1")))
    sent = next(call[2] for call in api.calls if call[0] == "test/pos/3")["messages"][-1]["content"][0]["text"]
    assert runner.TRUNCATION_MARKER in sent and "the ending" in sent


def test_custom_ids_are_reversible_and_api_safe():
    custom_id = runner.encode_custom_id("test/pos/12345_10")
    assert runner.CUSTOM_ID_PATTERN.fullmatch(custom_id)
    assert base64.urlsafe_b64decode(custom_id + "=" * (-len(custom_id) % 4)).decode() == "test/pos/12345_10"


# --- Batch mode ---------------------------------------------------------------------------------
BATCH_OUTCOMES = {
    "test/neg/1": {"type": "succeeded", "message": message("negative")},
    "test/neg/2": {"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error",
                                                                           "message": "bad request"}}},
    "test/pos/3": {"type": "succeeded", "message": message("positive")},
}  # test/pos/4 is not returned


def test_batch_flow_records_results_and_completion_time(tmp_path, eval_csv, clock):
    api = FakeBatchAPI(BATCH_OUTCOMES)
    report = execute(tmp_path, api, clock, mode="batch", poll_interval_s=30.0)
    assert [call[0] for call in api.calls].count("POST") == 1 and api.polls == 2
    assert clock.sleeps == [30.0]
    (created,) = api.created
    assert [r["custom_id"] for r in created["requests"]] == [runner.encode_custom_id(i) for i, _ in EVAL]
    params = created["requests"][0]["params"]
    assert params["model"] == HAIKU and "stream" not in params and params["output_config"]["format"]["type"] == "json_schema"
    records = {r["example_id"]: r for r in records_of(report)}
    assert {i: (r["parse_status"], r["batch_result_type"]) for i, r in records.items()} == {
        "test/neg/1": ("ok", "succeeded"), "test/neg/2": ("error", "errored"),
        "test/pos/3": ("ok", "succeeded"), "test/pos/4": ("missing", "not_returned")}
    ok_record = records["test/neg/1"]
    assert ok_record["batch_id"] == "msgbatch_test" and ok_record["batch_completion_s"] == 1800.0
    assert (ok_record["batch_created_utc"], ok_record["batch_ended_utc"]) == ("2026-11-08T12:00:00Z", "2026-11-08T12:30:00Z")
    assert ok_record["latency_s"] is None and ok_record["http_status"] is None
    assert ok_record["cost_usd"] == pytest.approx(HAIKU_COST / 2) and ok_record["batch_discount_applied"] is True
    assert records["test/neg/2"]["error_type"] == "invalid_request_error"
    events, _ = runner.read_jsonl(tmp_path / "results" / "test-run" / "batches.jsonl")
    assert [e["event"] for e in events] == ["submitted", "collected"] and events[1]["completion_s"] == 1800.0
    document = runner.summarize([report.records_path], eval_csv)
    (group,) = document["groups"]
    assert group["batches"]["n_batches"] == 1 and group["batches"]["per_batch"][0]["completion_s"] == 1800.0
    assert "latency_s" not in group
    again = FakeBatchAPI({"test/pos/4": {"type": "succeeded", "message": message("positive")}}, polls_until_end=1)
    report = execute(tmp_path, again, clock, mode="batch")
    assert [r["custom_id"] for r in again.created[0]["requests"]] == [runner.encode_custom_id("test/pos/4")]
    assert terminal_counts(report) == Counter({example_id: 1 for example_id, _ in EVAL})


def test_batch_resume_polls_the_submitted_batch_instead_of_resubmitting(tmp_path, eval_csv, clock):
    crashed = FakeBatchAPI(BATCH_OUTCOMES, crash_on_poll=True)
    with pytest.raises(RuntimeError, match="simulated crash"):
        execute(tmp_path, crashed, clock, mode="batch")
    assert len(crashed.created) == 1
    resumed = FakeBatchAPI(BATCH_OUTCOMES, polls_until_end=1)
    report = execute(tmp_path, resumed, clock, mode="batch")
    assert resumed.created == [] and resumed.polls == 1
    assert len(records_of(report)) == 4


# --- Summary and secrets ------------------------------------------------------------------------
def test_summary_counts_non_ok_as_incorrect_and_states_denominators(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({
        "test/neg/2": [ok("positive")],  # wrong label
        "test/pos/3": [httpx.Response(200, json=message(None, stop_reason="refusal"))],
        "test/pos/4": [api_error(500, "api_error")],  # missing after max_attempts=1
    })
    report = execute(tmp_path, api, clock, max_attempts=1)
    document = runner.summarize([report.records_path], eval_csv)
    (group,) = document["groups"]  # no group, and no numbers, for models without records
    assert (group["model"], group["mode"], group["n_eligible"]) == (HAIKU, "online", 4)
    assert group["n_by_final_status"] == {"ok": 2, "malformed": 0, "refusal": 1, "max_tokens": 0, "error": 0,
                                          "missing": 1, "not_attempted": 0}
    assert group["accuracy_all_eligible"] == {"value": 0.25, "numerator": 1, "denominator": 4}
    assert group["accuracy_parsed_ok"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    rates = group["error_rate_by_true_class"]
    assert rates["positive"] == {"all_eligible": {"value": 1.0, "numerator": 2, "denominator": 2},
                                 "parsed_ok": {"value": None, "numerator": 0, "denominator": 0}}
    assert rates["negative"] == {"all_eligible": {"value": 0.5, "numerator": 1, "denominator": 2},
                                 "parsed_ok": {"value": 0.5, "numerator": 1, "denominator": 2}}
    assert group["tokens"]["total"]["input_tokens"] == 3 * 400 and group["tokens"]["n_examples_with_usage"] == 3
    assert group["cost_usd"]["total"] == pytest.approx(3 * HAIKU_COST)
    assert group["latency_s"]["n_requests"] == 3 and group["latency_s"]["median"] == pytest.approx(0.25)
    assert "accuracy_all_eligible" in document["definitions"] and "batches" not in group
    with (report.records_path.parent / "per_example.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["example_id"], r["final_status"], r["correct"]) for r in rows] == [
        ("test/neg/1", "ok", "1"), ("test/neg/2", "ok", "0"), ("test/pos/3", "refusal", "0"),
        ("test/pos/4", "missing", "0")]
    out_dir = tmp_path / "cli-summary"
    assert runner.main(["summarize", "--records", str(report.records_path), "--examples", str(eval_csv),
                        "--out-dir", str(out_dir)]) == 0
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["groups"] == document["groups"]


def test_api_key_never_reaches_written_files(tmp_path, eval_csv, clock):
    api = FakeMessagesAPI({"test/neg/1": [api_error(400, "invalid_request_error", text=f"echo {DUMMY_KEY}")]})
    report = execute(tmp_path, api, clock)
    assert "[REDACTED]" in attempts_for(report, "test/neg/1")[0]["raw_text"]
    execute(tmp_path, FakeBatchAPI(BATCH_OUTCOMES), clock, mode="batch")
    runner.summarize([report.records_path], eval_csv)
    written = [path for path in (tmp_path / "results").rglob("*") if path.is_file()]
    assert {path.name for path in written} >= {"requests.jsonl", "run.json", "batches.jsonl", "summary.json",
                                               "per_example.csv"}
    assert all(DUMMY_KEY not in path.read_text(encoding="utf-8") for path in written)
