"""Claude zero- and few-shot sentiment baseline: reproducible, resumable and cost-bounded runs.

    python -m study.llm.runner run --examples EVAL.csv [--exemplars SHOTS.csv] [--mode online|batch]
    python -m study.llm.runner run ... --live --max-spend-usd 20
    python -m study.llm.runner summarize --records results/llm/RUN_ID/requests.jsonl --examples EVAL.csv

Without --live, `run` is a dry run: it builds and validates every request body and prints the plan
(example counts, token and cost bounds per model) without network access or file writes. Live runs
call the Messages API (online mode) or the Message Batches API (batch mode) over HTTPS with httpx
and append one record per attempt to requests.jsonl, so an interrupted run resumes where it stopped.
API facts (model ids, request and response shapes, retry semantics, prices) follow the claude-api
skill, read on 2026-11-08. study/llm/README.md documents the design and its limitations.
"""
import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import string
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TextIO

import httpx
import numpy as np

# --- Claude API facts (claude-api skill, read on 2026-11-08) -----------------------------------
API_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2026-10-21"  # anthropic-version header; this version rejects temperature/top_p/top_k
API_KEY_ENV = "ANTHROPIC_API_KEY"
MESSAGES_PATH = "/v1/messages"
BATCHES_PATH = "/v1/messages/batches"
DEFAULT_MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")  # cheapest first
#: Model-specific request fields, merged into every request body; the runner has no other
#: per-model branches. No sampling parameter is sent, because API version 2026-10-21 rejects
#: temperature, top_p and top_k on every model. `thinking` stays at each model's default
#: (adaptive), because Claude Opus 5.5 and Claude Sonnet 5.5 reject {"type": "disabled"}; thinking
#: tokens count toward max_tokens and are billed as output. Effort "low" is the skill's starting
#: point for classification.
MODEL_REQUEST_OPTIONS: dict[str, dict[str, Any]] = {
    "claude-haiku-5-5": {"output_config": {"effort": "low"}},
    "claude-sonnet-5-5": {"output_config": {"effort": "low"}},
    "claude-opus-5-5": {"output_config": {"effort": "low"}},
    "claude-fable-5-1": {"output_config": {"effort": "low"}},
}
SAMPLING_PARAMETERS = ("temperature", "top_p", "top_k")
MAX_CACHE_BREAKPOINTS = 4
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})
#: Rejected before processing: the only statuses on which a batch creation is resent.
CREATE_RETRY_STATUS = frozenset({429, 529})
#: Authentication, billing, permission and unknown-model errors concern the run, not an example.
ABORT_STATUS = frozenset({401, 402, 403, 404})
#: Errored batch items of these types are example-level and final; other types stay open.
BATCH_EXAMPLE_ERROR_TYPES = frozenset({"invalid_request_error", "request_too_large"})
BATCH_MAX_REQUESTS = 100_000
BATCH_MAX_BYTES = 200 * 1024 * 1024  # headroom below the documented 256 MB per batch
CUSTOM_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

# --- Local conventions --------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
PRICING_PATH = PACKAGE_DIR / "pricing.json"
DEFAULT_RESULTS_DIR = Path("results") / "llm"
RECORDS_NAME, MANIFEST_NAME, BATCH_LOG_NAME = "requests.jsonl", "run.json", "batches.jsonl"
PROMPT_FILES = ("system.txt", "user.txt", "response_schema.json")
LABEL_TO_INT = {"negative": 0, "positive": 1}
INT_TO_LABEL = {value: name for name, value in LABEL_TO_INT.items()}
TERMINAL_STATUSES = frozenset({"ok", "malformed", "refusal", "max_tokens", "error"})
FINAL_STATUSES = ("ok", "malformed", "refusal", "max_tokens", "error", "missing", "not_attempted")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
RATE_FIELDS = ("input", "output", "cache_write_5m", "cache_write_1h", "cache_read")
RAW_TEXT_LIMIT = 2_000
TRUNCATION_MARKER = "\n[... middle of review omitted ...]\n"
MIN_MAX_REVIEW_CHARS = 100
CHARS_PER_TOKEN = 3  # pre-flight estimate: input tokens = ceil(characters / 3)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,100}")
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_POLL_INTERVAL_S = 60.0
DEFAULT_BATCH_MAX_WAIT_S = 24 * 3600.0
MAX_CONCURRENCY = 16
PROGRESS_EVERY = 250


class RunnerError(Exception):
    """An expected failure that the command line reports without a traceback."""


class InputError(RunnerError):
    """Invalid examples, exemplars, prompt files, pricing table or run configuration."""


class PreflightRefused(RunnerError):
    """The conservative cost bound of the planned requests exceeds --max-spend-usd."""


class RunAborted(RunnerError):
    """A run-level API failure: credentials, permissions, unknown model or a failed batch call."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


@dataclass(frozen=True)
class Clock:
    """Time sources, injectable so that tests neither wait nor depend on the wall clock."""

    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.perf_counter
    utc_now: Callable[[], datetime] = _utc_now


# --- Examples and exemplars ---------------------------------------------------------------------
@dataclass(frozen=True)
class Example:
    """A labelled review: label 0 is negative, 1 is positive."""

    example_id: str
    text: str
    label: int


def load_examples(path: Path) -> list[Example]:
    """Read a CSV (plain or .gz) with columns example_id, text and label; ids must be unique."""
    if not path.is_file():
        raise InputError(f"{path}: file not found")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        absent = {"example_id", "text", "label"} - set(reader.fieldnames or ())
        if absent:
            raise InputError(f"{path}: missing column(s) {sorted(absent)}")
        examples: list[Example] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=1):
            example_id, text, label = row["example_id"], row["text"], row["label"]
            if not example_id or example_id in seen:
                raise InputError(f"{path}: data row {row_number}: empty or duplicate example_id {example_id!r}")
            if label not in ("0", "1"):
                raise InputError(f"{path}: data row {row_number}: label must be 0 or 1, got {label!r}")
            if not text or not text.strip():
                raise InputError(f"{path}: data row {row_number}: empty text")
            seen.add(example_id)
            examples.append(Example(example_id, text, int(label)))
    if not examples:
        raise InputError(f"{path}: no examples")
    return examples


def _normalise(text: str) -> str:
    """Comparison key for exemplar/evaluation overlap: whitespace collapsed, casefolded."""
    return " ".join(text.split()).casefold()


def validate_exemplars(exemplars: Sequence[Example], examples: Sequence[Example]) -> None:
    """Require training-partition exemplars that share neither id nor text with the evaluation set."""
    problems = []
    outside = [e.example_id for e in exemplars if not e.example_id.startswith("train/")]
    if outside:
        problems.append(f"ids outside the training partition (must start with 'train/'): {outside[:5]}")
    eval_ids = {e.example_id for e in examples}
    shared_ids = [e.example_id for e in exemplars if e.example_id in eval_ids]
    if shared_ids:
        problems.append(f"ids also among the evaluation examples: {shared_ids[:5]}")
    eval_texts = {_normalise(e.text) for e in examples}
    shared_texts = [e.example_id for e in exemplars if _normalise(e.text) in eval_texts]
    if shared_texts:
        problems.append(f"texts also among the evaluation examples: {shared_texts[:5]}")
    if problems:
        raise InputError("invalid exemplars: " + "; ".join(problems))


def example_set_sha256(examples: Sequence[Example]) -> str:
    """sha256 over the sorted example ids, each paired with the sha256 of its text."""
    ordered = sorted(examples, key=lambda e: e.example_id)
    return sha256_text("".join(f"{e.example_id}\t{sha256_text(e.text)}\n" for e in ordered))


def exemplar_set_sha256(exemplars: Sequence[Example]) -> str:
    """sha256 over the exemplars in prompt order: id, text sha256 and label."""
    return sha256_text("".join(f"{e.example_id}\t{sha256_text(e.text)}\t{e.label}\n" for e in exemplars))


# --- Versioned prompts and the response schema --------------------------------------------------
_SCHEMA_KEYWORDS = frozenset({"type", "properties", "required", "additionalProperties", "enum", "description", "title"})
_SCHEMA_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,), "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "null": (type(None),),
}


def _check_schema(schema: Any, where: str = "$") -> None:
    """Accept the JSON Schema subset that schema_violations() evaluates; objects must be closed."""
    if not isinstance(schema, dict):
        raise InputError(f"response schema {where}: expected an object")
    unsupported = set(schema) - _SCHEMA_KEYWORDS
    if unsupported:
        raise InputError(f"response schema {where}: unsupported keyword(s) {sorted(unsupported)}")
    kind = schema.get("type")
    if kind is not None and kind not in _SCHEMA_TYPES:
        raise InputError(f"response schema {where}: unsupported type {kind!r}")
    if kind == "object":
        if schema.get("additionalProperties") is not False:
            raise InputError(f"response schema {where}: structured outputs need additionalProperties: false")
        for name, subschema in schema.get("properties", {}).items():
            _check_schema(subschema, f"{where}.{name}")


def schema_violations(value: Any, schema: Mapping[str, Any], where: str = "$") -> list[str]:
    """Problems with `value` under a schema accepted by _check_schema (empty when it conforms)."""
    kind = schema.get("type")
    if kind is not None and (not isinstance(value, _SCHEMA_TYPES[kind])
                             or (isinstance(value, bool) and kind in ("integer", "number"))):
        return [f"{where}: expected {kind}"]
    problems = []
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        problems += [f"{where}: missing {key!r}" for key in schema.get("required", ()) if key not in value]
        if schema.get("additionalProperties") is False:
            problems += [f"{where}: unexpected {key!r}" for key in value if key not in properties]
        for key, subschema in properties.items():
            if key in value:
                problems += schema_violations(value[key], subschema, f"{where}.{key}")
    return problems


@dataclass(frozen=True)
class PromptSpec:
    """A prompt version: system text, user template with one {review} slot, response schema."""

    version: str
    system: str
    user_template: str
    schema: dict[str, Any]
    file_sha256: dict[str, str]
    sha256: str  # over the three file hashes: the identity recorded with every request

    def render_user(self, review: str) -> str:
        return self.user_template.format(review=review)


def load_prompt(version: str, prompts_dir: Path = PROMPTS_DIR) -> PromptSpec:
    """Load and check prompts/<version>/; file hashes are taken over LF-normalised text."""
    directory = prompts_dir / version
    texts = {}
    for name in PROMPT_FILES:
        path = directory / name
        if not path.is_file():
            raise InputError(f"prompt version {version!r}: {path} not found")
        texts[name] = path.read_text(encoding="utf-8")  # universal newlines: a CRLF checkout hashes as LF
    template = texts["user.txt"].strip()
    try:
        fields = [(name, spec, conversion) for _, name, spec, conversion in string.Formatter().parse(template)
                  if name is not None]
    except ValueError as exc:
        raise InputError(f"{directory / 'user.txt'}: {exc}") from exc
    if fields != [("review", "", None)]:
        raise InputError(f"{directory / 'user.txt'}: the template needs exactly one plain {{review}} field")
    try:
        schema = json.loads(texts["response_schema.json"])
    except ValueError as exc:
        raise InputError(f"{directory / 'response_schema.json'}: {exc}") from exc
    _check_schema(schema)
    label = schema.get("properties", {}).get("label", {})
    if "label" not in schema.get("required", ()) or set(label.get("enum", ())) != set(LABEL_TO_INT):
        raise InputError(f"{directory / 'response_schema.json'}: needs a required 'label' with enum {sorted(LABEL_TO_INT)}")
    file_sha256 = {name: sha256_text(texts[name]) for name in PROMPT_FILES}
    return PromptSpec(version, texts["system.txt"].strip(), template, schema, file_sha256, _canonical_sha256(file_sha256))


# --- Truncation and request bodies --------------------------------------------------------------
@dataclass(frozen=True)
class PromptInput:
    """An example as it enters a prompt, after the truncation policy."""

    example: Example
    text: str
    truncated: bool

    @property
    def original_chars(self) -> int:
        return len(self.example.text)


def truncate_review(text: str, max_chars: int | None) -> tuple[str, bool]:
    """Truncation policy: keep the head and the tail of an over-long review around a fixed marker.

    Deterministic. The result has at most `max_chars` characters (Unicode code points); the head
    keeps the extra character when the remaining budget is odd. `None` disables truncation.
    """
    if max_chars is None or len(text) <= max_chars:
        return text, False
    budget = max_chars - len(TRUNCATION_MARKER)
    head = (budget + 1) // 2
    return text[:head] + TRUNCATION_MARKER + text[len(text) - (budget - head):], True


def prepare(example: Example, max_chars: int | None) -> PromptInput:
    text, truncated = truncate_review(example.text, max_chars)
    return PromptInput(example, text, truncated)


def _deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _cache_control(ttl: str) -> dict[str, str] | None:
    """Breakpoint for the end of the shared prefix; None when caching is off."""
    if ttl == "off":
        return None
    return {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": ttl}


@dataclass(frozen=True)
class RequestBuilder:
    """Builds Messages API bodies whose shared prefix (system prompt, exemplars) is byte-identical."""

    prompt: PromptSpec
    exemplars: tuple[PromptInput, ...]
    max_tokens: int
    cache_ttl: str

    def body(self, model: str, review: str) -> dict[str, Any]:
        """Request body for one review: exemplars as prior user/assistant turns, then the review.

        The cache breakpoint sits on the last block of the shared prefix (the last exemplar answer,
        or the system prompt in a zero-shot run), so the varying review follows it.
        """
        system = {"type": "text", "text": self.prompt.system}
        messages: list[dict[str, Any]] = []
        for shot in self.exemplars:
            answer = json.dumps({"label": INT_TO_LABEL[shot.example.label]})
            messages.append({"role": "user", "content": [{"type": "text", "text": self.prompt.render_user(shot.text)}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        marker = _cache_control(self.cache_ttl)
        if marker is not None:
            (messages[-1]["content"][-1] if messages else system)["cache_control"] = marker
        messages.append({"role": "user", "content": [{"type": "text", "text": self.prompt.render_user(review)}]})
        body = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": [system],
            "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": self.prompt.schema}},
        }
        return _deep_merge(body, MODEL_REQUEST_OPTIONS[model])


def validate_request_body(body: Mapping[str, Any]) -> list[str]:
    """Static checks against the request constraints the claude-api skill documents (empty if valid)."""
    problems = []
    if not isinstance(body.get("model"), str) or not body["model"]:
        problems.append("model must be a non-empty string")
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        problems.append("max_tokens must be a positive integer")
    sampling = [name for name in SAMPLING_PARAMETERS if name in body]
    if sampling:
        problems.append(f"{', '.join(sampling)} rejected on API version {API_VERSION}")
    messages = body.get("messages") or []
    roles = [message.get("role") for message in messages]
    if not roles or roles[0] != "user" or roles[-1] != "user":
        problems.append("messages must start and end with a user turn (assistant prefill is rejected)")
    if not set(roles) <= {"user", "assistant"} or any(first == second for first, second in zip(roles, roles[1:])):
        problems.append("message roles must alternate between user and assistant")
    blocks = list(body.get("system") or []) + [block for message in messages for block in message.get("content") or []]
    if any(block.get("type") != "text" or not isinstance(block.get("text"), str) or not block["text"].strip()
           for block in blocks):
        problems.append("every system and message block must be a non-empty text block")
    if sum("cache_control" in block for block in blocks) > MAX_CACHE_BREAKPOINTS:
        problems.append(f"more than {MAX_CACHE_BREAKPOINTS} cache_control breakpoints")
    output_format = (body.get("output_config") or {}).get("format") or {}
    if output_format.get("type") != "json_schema" or not isinstance(output_format.get("schema"), dict):
        problems.append("output_config.format must be a json_schema structured-output format")
    return problems


def estimate_input_tokens(body: Mapping[str, Any]) -> int:
    """Pre-flight input estimate: ceil(characters / 3) over system text, message texts and the schema."""
    texts = [block["text"] for block in body["system"]]
    texts += [block["text"] for message in body["messages"] for block in message["content"]]
    characters = sum(map(len, texts)) + len(json.dumps(body["output_config"]["format"]["schema"]))
    return math.ceil(characters / CHARS_PER_TOKEN)


def encode_custom_id(example_id: str) -> str:
    """Batch custom_id: unpadded URL-safe base64 of the example id (reversible, [A-Za-z0-9_-])."""
    return base64.urlsafe_b64encode(example_id.encode("utf-8")).decode("ascii").rstrip("=")


# --- Prices -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RateCard:
    """USD per million tokens for each billed token class."""

    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


@dataclass(frozen=True)
class ModelPrices:
    """A model's standard rate card and, where one exists, the card for long prompts."""

    standard: RateCard
    long_context: RateCard | None = None
    long_context_above: int | None = None  # prompt tokens above which the long card applies

    def card(self, prompt_tokens: int) -> RateCard:
        if self.long_context is not None and self.long_context_above is not None \
                and prompt_tokens > self.long_context_above:
            return self.long_context
        return self.standard


def _usage_count(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


@dataclass(frozen=True)
class Pricing:
    """pricing.json: per-model rate cards and the Message Batches multiplier."""

    source: str
    source_date: str
    batch_multiplier: float
    models: dict[str, ModelPrices]
    sha256: str

    def cost_usd(self, model: str, usage: Mapping[str, Any], *, batch: bool, cache_ttl: str) -> float:
        """Cost of one response from its usage block (thinking tokens are part of output_tokens).

        Cache writes are priced per TTL from usage.cache_creation when present; writes it does not
        break down are priced at the run's TTL. A long-prompt rate card applies to the whole request.
        """
        fresh, output = _usage_count(usage, "input_tokens"), _usage_count(usage, "output_tokens")
        written = _usage_count(usage, "cache_creation_input_tokens")
        read = _usage_count(usage, "cache_read_input_tokens")
        split = usage.get("cache_creation")
        split = split if isinstance(split, Mapping) else {}
        written_5m = _usage_count(split, "ephemeral_5m_input_tokens")
        written_1h = _usage_count(split, "ephemeral_1h_input_tokens")
        unsplit = max(written - written_5m - written_1h, 0)
        if cache_ttl == "1h":
            written_1h += unsplit
        else:
            written_5m += unsplit
        card = self.models[model].card(fresh + written + read)
        dollars = (fresh * card.input + written_5m * card.cache_write_5m + written_1h * card.cache_write_1h
                   + read * card.cache_read + output * card.output) / 1_000_000
        return dollars * (self.batch_multiplier if batch else 1.0)

    def request_bound_usd(self, model: str, input_tokens: int, max_tokens: int, *, batch: bool,
                          cache_ttl: str) -> float:
        """Upper bound for one request: every input token at the higher of the input and cache-write
        rates (no cache-read discount) and max_tokens output tokens."""
        card = self.models[model].card(input_tokens)
        input_rate = card.input
        if cache_ttl != "off":
            input_rate = max(input_rate, card.cache_write_1h if cache_ttl == "1h" else card.cache_write_5m)
        dollars = (input_tokens * input_rate + max_tokens * card.output) / 1_000_000
        return dollars * (self.batch_multiplier if batch else 1.0)


def load_pricing(path: Path = PRICING_PATH) -> Pricing:
    """Read pricing.json; its `source` fields record where and when the prices were taken."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        models = {}
        for model, entry in data["models"].items():
            long_entry = entry.get("long_context")
            models[model] = ModelPrices(
                standard=RateCard(**{name: float(entry[name]) for name in RATE_FIELDS}),
                long_context=RateCard(**{name: float(long_entry[name]) for name in RATE_FIELDS}) if long_entry else None,
                long_context_above=int(long_entry["applies_above_prompt_tokens"]) if long_entry else None,
            )
        return Pricing(str(data["source"]), str(data["source_date"]), float(data["batch_multiplier"]), models,
                       sha256_text(text))
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"{path}: malformed pricing table ({exc!r})") from exc


# --- Responses ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedResponse:
    """Classification of one Messages API response: status, label (0/1 when ok) and text output."""

    status: str
    label: int | None
    text: str


def parse_message(message: Mapping[str, Any], schema: Mapping[str, Any]) -> ParsedResponse:
    """Classify a Messages API response body: by stop_reason first, then by its structured output.

    Text blocks are read by type (thinking blocks, which precede them, are skipped). A refusal or
    max_tokens stop is reported as such whatever the content; any stop reason other than end_turn,
    unparseable JSON or a schema violation is `malformed`.
    """
    content = message.get("content")
    blocks = content if isinstance(content, list) else []
    text = "".join(str(block.get("text", "")) for block in blocks
                   if isinstance(block, Mapping) and block.get("type") == "text")
    stop_reason = message.get("stop_reason")
    if stop_reason in ("refusal", "max_tokens"):
        return ParsedResponse(stop_reason, None, text)
    if stop_reason != "end_turn":
        return ParsedResponse("malformed", None, text)
    try:
        value = json.loads(text)
    except ValueError:
        return ParsedResponse("malformed", None, text)
    if schema_violations(value, schema):
        return ParsedResponse("malformed", None, text)
    return ParsedResponse("ok", LABEL_TO_INT[value["label"]], text)


# --- HTTP ---------------------------------------------------------------------------------------
def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Seconds to wait from a retry-after header (delta-seconds or HTTP-date); None if absent or invalid."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            moment = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        seconds = (moment - now).total_seconds()
    return max(0.0, seconds) if math.isfinite(seconds) else None


@dataclass(frozen=True)
class HttpAttempt:
    """One HTTP exchange, or a transport failure (status None)."""

    started_utc: str
    latency_s: float
    status: int | None
    request_id: str | None
    payload: Any
    text: str
    retry_after_s: float | None = None
    error_type: str | None = None
    ambiguous: bool = False  # transport failure after which the server may have acted on the request

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in RETRYABLE_STATUS

    @property
    def succeeded(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    def describe(self) -> str:
        what = f"HTTP {self.status}" if self.status is not None else "network error"
        return f"{what} ({self.error_type or 'no error type'}): {self.text[:300]}"


_UNSENT_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class AnthropicHTTP:
    """Minimal REST client for the Messages and Message Batches endpoints.

    The API key is read from ANTHROPIC_API_KEY only when a request is sent, travels only in the
    x-api-key header, and is replaced by [REDACTED] in every response text this client returns.
    """

    def __init__(self, transport: httpx.BaseTransport | None, timeout_s: float, clock: Clock) -> None:
        self._http = httpx.Client(base_url=API_BASE_URL, transport=transport, timeout=timeout_s)
        self._clock = clock

    def close(self) -> None:
        self._http.close()

    def request(self, method: str, path: str, *, json_body: Mapping[str, Any] | None = None,
                content: bytes | None = None) -> HttpAttempt:
        api_key = os.environ.get(API_KEY_ENV)
        if not api_key:
            raise RunAborted(f"{API_KEY_ENV} is not set; live runs read the API key from the environment")
        headers = {"x-api-key": api_key, "anthropic-version": API_VERSION}
        if content is not None:
            headers["content-type"] = "application/json"
        started, t0 = _iso(self._clock.utc_now()), self._clock.monotonic()
        try:
            response = self._http.request(method, path, json=json_body, content=content, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            return HttpAttempt(started, self._clock.monotonic() - t0, None, None, None,
                               f"{type(exc).__name__}: {exc}".replace(api_key, "[REDACTED]"),
                               error_type=type(exc).__name__,
                               ambiguous=not isinstance(exc, _UNSENT_TRANSPORT_ERRORS))
        latency = self._clock.monotonic() - t0
        text = response.text.replace(api_key, "[REDACTED]")
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        request_id = response.headers.get("request-id") or (payload.get("request_id") if isinstance(payload, dict) else None)
        return HttpAttempt(started, latency, response.status_code, request_id, payload, text,
                           retry_after_s=parse_retry_after(response.headers.get("retry-after"), self._clock.utc_now()),
                           error_type=error.get("type") if isinstance(error, dict) else None)


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter; a retry-after header, when present, sets the wait instead."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0

    def delay_s(self, attempt: int, retry_after_s: float | None, rng: random.Random) -> float:
        if retry_after_s is not None:
            return retry_after_s
        return min(self.base_delay_s * 2 ** (attempt - 1) + rng.uniform(0.0, 1.0), self.max_delay_s)


# --- Records, manifests and spending ------------------------------------------------------------
class JsonlWriter:
    """Append-only JSONL writer: every record is flushed and fsynced before write() returns."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        unterminated = path.is_file() and path.stat().st_size > 0 and not _ends_with_newline(path)
        self._handle = path.open("ab")
        self._lock = threading.Lock()
        if unterminated:  # an interrupted write left a partial line; read_jsonl skips it
            self._append(b"\n")

    def _append(self, data: bytes) -> None:
        self._handle.write(data)
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self._append(line.encode("utf-8"))

    def close(self) -> None:
        self._handle.close()


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Records of a JSONL file (none if it is absent) and the number of unreadable lines skipped."""
    if not path.is_file():
        return [], 0
    records, skipped = [], 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                skipped += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                skipped += 1
    return records, skipped


@dataclass(frozen=True, kw_only=True)
class RequestRecord:
    """One line of requests.jsonl: one attempt for one example, or a not-attempted marker."""

    run_id: str
    mode: str
    model: str
    prompt_version: str
    prompt_sha256: str
    example_id: str
    attempt: int | None
    request_started_utc: str | None = None
    latency_s: float | None = None  # online only: wall time of the HTTP request
    http_status: int | None = None
    request_id: str | None = None
    stop_reason: str | None = None
    stop_details: Any = None
    usage: dict[str, Any] | None = None  # verbatim from the response
    cost_usd: float = 0.0
    batch_discount_applied: bool = False
    parsed_label: int | None = None  # 0 negative, 1 positive; set only when parse_status is ok
    parse_status: str
    truncated: bool
    original_chars: int
    raw_text: str | None = None  # model text output, or the error body; at most RAW_TEXT_LIMIT chars
    error_type: str | None = None
    batch_id: str | None = None
    batch_created_utc: str | None = None
    batch_ended_utc: str | None = None
    batch_completion_s: float | None = None  # whole-batch completion time, not a per-request latency
    batch_result_type: str | None = None


class SpendLedger:
    """Thread-safe accounting of this invocation's spend against --max-spend-usd."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.realized_usd = 0.0
        self.exhausted = False
        self._reserved_usd = 0.0
        self._lock = threading.Lock()

    def reserve(self, bound_usd: float) -> bool:
        """Reserve a request's upper bound; refuse, and stay exhausted, once it would exceed the limit."""
        with self._lock:
            if self.exhausted or self.realized_usd + self._reserved_usd + bound_usd > self.limit_usd:
                self.exhausted = True
                return False
            self._reserved_usd += bound_usd
            return True

    def settle(self, bound_usd: float, cost_usd: float) -> None:
        with self._lock:
            self._reserved_usd -= bound_usd
            self.realized_usd += cost_usd


# --- Running ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSettings:
    """Configuration of one `run` invocation; the command-line flags map onto these fields."""

    examples_path: Path
    exemplars_path: Path | None = None
    models: tuple[str, ...] = DEFAULT_MODELS
    mode: str = "online"
    prompt_version: str = "v1"
    max_review_chars: int | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    cache_ttl: str = "5m"
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    concurrency: int = 1
    run_id: str | None = None
    results_dir: Path = DEFAULT_RESULTS_DIR
    records_path: Path | None = None
    live: bool = False
    max_spend_usd: float | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    batch_max_wait_s: float = DEFAULT_BATCH_MAX_WAIT_S
    prompts_dir: Path = PROMPTS_DIR
    pricing_path: Path = PRICING_PATH


def _check_settings(settings: RunSettings) -> None:
    problems = []
    if settings.mode not in ("online", "batch"):
        problems.append("mode must be online or batch")
    if settings.cache_ttl not in ("5m", "1h", "off"):
        problems.append("cache TTL must be 5m, 1h or off")
    if not settings.models or len(set(settings.models)) != len(settings.models):
        problems.append("models must be a non-empty list without repeats")
    if settings.max_tokens < 1 or settings.max_attempts < 1:
        problems.append("max_tokens and max_attempts must be positive")
    if not 1 <= settings.concurrency <= MAX_CONCURRENCY:
        problems.append(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    if settings.max_review_chars is not None and settings.max_review_chars < MIN_MAX_REVIEW_CHARS:
        problems.append(f"max_review_chars must be at least {MIN_MAX_REVIEW_CHARS}")
    if settings.run_id is not None and not RUN_ID_PATTERN.fullmatch(settings.run_id):
        problems.append("run_id may contain only letters, digits, '.', '_' and '-'")
    if settings.live and (settings.max_spend_usd is None or settings.max_spend_usd <= 0):
        problems.append("live runs need a positive max_spend_usd")
    if settings.timeout_s <= 0 or settings.poll_interval_s <= 0 or settings.batch_max_wait_s < 0:
        problems.append("timeout, poll interval and batch wait must be positive")
    if problems:
        raise InputError("; ".join(problems))


@dataclass
class ModelPlan:
    """What one model still needs in this run, with conservative token and cost bounds."""

    model: str
    n_eligible: int
    n_done: int
    n_in_flight: int
    pending: list[PromptInput]
    bounds_usd: dict[str, float]
    input_tokens_estimate: int
    max_request_input_tokens: int
    output_tokens_bound: int

    @property
    def cost_bound_usd(self) -> float:
        return sum(self.bounds_usd.values())


@dataclass
class RunReport:
    """Outcome of one invocation: the plan and, for live runs, the records it wrote."""

    run_id: str
    records_path: Path
    live: bool
    plans: dict[str, ModelPlan]
    records_written: dict[str, dict[str, int]]
    realized_cost_usd: float
    spend_limit_reached: bool
    pending_batches: list[str]

    @property
    def cost_bound_usd(self) -> float:
        return sum(plan.cost_bound_usd for plan in self.plans.values())


@dataclass(frozen=True)
class _Outcome:
    """What one attempt produced, before it becomes a record (`retryable`, `aborted` are interim)."""

    status: str
    label: int | None = None
    text: str = ""
    usage: dict[str, Any] | None = None
    cost_usd: float = 0.0
    stop_reason: str | None = None
    stop_details: Any = None
    error_type: str | None = None


@dataclass(frozen=True)
class _Submission:
    """A submitted message batch; reserved_usd is None for a batch recovered from an earlier invocation."""

    batch_id: str
    model: str
    example_ids: tuple[str, ...]
    submitted_utc: str
    reserved_usd: float | None


@dataclass
class _OnlineState:
    """Coordination between the worker threads of one model's online pass."""

    halt: threading.Event = field(default_factory=threading.Event)
    sent: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_sent(self, example_id: str) -> None:
        with self.lock:
            self.sent.add(example_id)


class _Run:
    """One invocation of `run`: loads and checks the inputs, plans, then executes or reports."""

    def __init__(self, settings: RunSettings, clock: Clock, rng: random.Random, out: TextIO) -> None:
        _check_settings(settings)
        self.s, self.clock, self.rng, self.out = settings, clock, rng, out
        self.prompt = load_prompt(settings.prompt_version, settings.prompts_dir)
        self.pricing = load_pricing(settings.pricing_path)
        unknown = [m for m in settings.models if m not in self.pricing.models or m not in MODEL_REQUEST_OPTIONS]
        if unknown:
            raise InputError(f"unknown model(s) {unknown}; add each to pricing.json and MODEL_REQUEST_OPTIONS")
        examples = load_examples(settings.examples_path)
        exemplars = load_examples(settings.exemplars_path) if settings.exemplars_path else []
        validate_exemplars(exemplars, examples)
        self.inputs = [prepare(e, settings.max_review_chars) for e in sorted(examples, key=lambda e: e.example_id)]
        self.inputs_by_id = {item.example.example_id: item for item in self.inputs}
        shots = tuple(prepare(e, settings.max_review_chars) for e in exemplars)
        self.builder = RequestBuilder(self.prompt, shots, settings.max_tokens, settings.cache_ttl)
        self.retry = RetryPolicy(max_attempts=settings.max_attempts)
        self.config = {
            "anthropic_version": API_VERSION,
            "prompt_version": self.prompt.version,
            "prompt_sha256": self.prompt.sha256,
            "prompt_files_sha256": self.prompt.file_sha256,
            "eval_set_sha256": example_set_sha256(examples),
            "n_eval_examples": len(examples),
            "exemplar_set_sha256": exemplar_set_sha256(exemplars) if exemplars else None,
            "exemplar_ids": [e.example_id for e in exemplars],
            "max_review_chars": settings.max_review_chars,
            "truncation_marker": TRUNCATION_MARKER,
            "max_tokens": settings.max_tokens,
            "cache_ttl": settings.cache_ttl,
        }
        self.config_sha256 = _canonical_sha256(self.config)
        self.run_id = settings.run_id or f"{self.prompt.version}-{self.config_sha256[:12]}"
        self.records_path = settings.records_path or settings.results_dir / self.run_id / RECORDS_NAME
        self.manifest_path = self.records_path.parent / MANIFEST_NAME
        self.batch_log_path = self.records_path.parent / BATCH_LOG_NAME
        self.manifest = self._read_manifest()
        records, self.skipped_lines = read_jsonl(self.records_path)
        self.done: dict[str, set[str]] = defaultdict(set)  # model -> ids with a terminal record
        self.attempts: Counter[tuple[str, str]] = Counter()  # (model, id) -> highest attempt number
        for record in records:
            if (record.get("run_id"), record.get("mode"), record.get("prompt_version")) != \
                    (self.run_id, settings.mode, self.prompt.version):
                continue
            key = (str(record.get("model")), str(record.get("example_id")))
            if record.get("parse_status") in TERMINAL_STATUSES:
                self.done[key[0]].add(key[1])
            if isinstance(record.get("attempt"), int):
                self.attempts[key] = max(self.attempts[key], record["attempt"])
        self.recovered = self._uncollected_batches() if settings.mode == "batch" else []
        self.ledger = SpendLedger(settings.max_spend_usd if settings.max_spend_usd is not None else math.inf)
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.pending_batches: list[str] = []
        self._lock = threading.Lock()
        self.writer: JsonlWriter | None = None
        self.batch_log: JsonlWriter | None = None
        self.client: AnthropicHTTP | None = None

    # -- run bookkeeping
    def _read_manifest(self) -> dict[str, Any] | None:
        """The run's manifest if one exists, after checking that it matches this configuration."""
        if not self.manifest_path.is_file():
            return None
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        stored = manifest.get("config", {})
        differing = sorted(key for key in set(stored) | set(self.config) if stored.get(key) != self.config.get(key))
        if manifest.get("run_id") != self.run_id or differing:
            raise InputError(f"{self.manifest_path} belongs to a run with a different configuration "
                             f"({', '.join(differing) or 'run_id'}); choose another --run-id or --records")
        for model in self.s.models:
            stored_options = manifest.get("models", {}).get(model, {}).get("request_options")
            if stored_options is not None and stored_options != MODEL_REQUEST_OPTIONS[model]:
                raise InputError(f"{model}: request options differ from those recorded in {self.manifest_path}; "
                                 "choose another --run-id")
        return manifest

    def _write_manifest(self) -> None:
        now = _iso(self.clock.utc_now())
        manifest = self.manifest or {
            "run_id": self.run_id, "created_utc": now, "config_sha256": self.config_sha256, "config": self.config,
            "pricing": {"source": self.pricing.source, "source_date": self.pricing.source_date},
            "models": {}, "invocations": [],
        }
        for model in self.s.models:
            manifest["models"].setdefault(model, {"request_options": MODEL_REQUEST_OPTIONS[model], "first_used_utc": now})
        manifest["invocations"].append({
            "started_utc": now, "mode": self.s.mode, "models": list(self.s.models),
            "examples_path": str(self.s.examples_path),
            "exemplars_path": str(self.s.exemplars_path) if self.s.exemplars_path else None,
            "max_spend_usd": self.s.max_spend_usd, "max_attempts": self.s.max_attempts,
            "concurrency": self.s.concurrency, "pricing_sha256": self.pricing.sha256,
        })
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_name(MANIFEST_NAME + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.manifest_path)
        self.manifest = manifest

    def _uncollected_batches(self) -> list[_Submission]:
        events, _ = read_jsonl(self.batch_log_path)
        collected = {event.get("batch_id") for event in events if event.get("event") == "collected"}
        return [_Submission(e["batch_id"], e["model"], tuple(e["example_ids"]), e["submitted_utc"], None)
                for e in events
                if e.get("event") == "submitted" and e.get("run_id") == self.run_id
                and e.get("prompt_version") == self.prompt.version and e.get("batch_id") not in collected]

    def _say(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    # -- planning
    def plan(self) -> dict[str, ModelPlan]:
        """Pending examples, validated request bodies and cost bounds for every model."""
        batch = self.s.mode == "batch"
        plans = {}
        for model in self.s.models:
            in_flight = {eid for sub in self.recovered if sub.model == model for eid in sub.example_ids}
            done = self.done.get(model, set())
            pending = [item for item in self.inputs
                       if item.example.example_id not in done and item.example.example_id not in in_flight]
            bounds: dict[str, float] = {}
            estimates = []
            for item in pending:
                body = self.builder.body(model, item.text)
                problems = validate_request_body(body)
                if batch and not CUSTOM_ID_PATTERN.fullmatch(encode_custom_id(item.example.example_id)):
                    problems.append("example_id too long for a batch custom_id (at most 48 bytes)")
                if problems:
                    raise InputError(f"{model}, {item.example.example_id}: invalid request: {'; '.join(problems)}")
                estimate = estimate_input_tokens(body)
                estimates.append(estimate)
                bounds[item.example.example_id] = self.pricing.request_bound_usd(
                    model, estimate, self.s.max_tokens, batch=batch, cache_ttl=self.s.cache_ttl)
            plans[model] = ModelPlan(
                model=model, n_eligible=len(self.inputs), n_done=len(done & self.inputs_by_id.keys()),
                n_in_flight=len(in_flight), pending=pending, bounds_usd=bounds,
                input_tokens_estimate=sum(estimates), max_request_input_tokens=max(estimates, default=0),
                output_tokens_bound=len(pending) * self.s.max_tokens)
        return plans

    def _print_plan(self, plans: Mapping[str, ModelPlan], bound_usd: float) -> None:
        heading = "Live run" if self.s.live else "Dry run (no network access, no files written)"
        self._say(f"{heading}: run {self.run_id}, mode {self.s.mode}, prompt {self.prompt.version} "
                  f"(sha256 {self.prompt.sha256[:12]}), API version {API_VERSION}")
        self._say(f"eligible examples {len(self.inputs)} (set sha256 {self.config['eval_set_sha256'][:12]}), "
                  f"exemplars {len(self.builder.exemplars)}, max_tokens {self.s.max_tokens}, "
                  f"cache {self.s.cache_ttl}, truncation {self.s.max_review_chars or 'off'}")
        self._say(f"records: {self.records_path}")
        self._say(f"{'model':<20}{'eligible':>9}{'done':>7}{'in flight':>10}{'pending':>9}"
                  f"{'est. input tokens':>19}{'max output tokens':>19}{'cost bound USD':>16}")
        for plan in plans.values():
            self._say(f"{plan.model:<20}{plan.n_eligible:>9}{plan.n_done:>7}{plan.n_in_flight:>10}"
                      f"{len(plan.pending):>9}{plan.input_tokens_estimate:>19}{plan.output_tokens_bound:>19}"
                      f"{plan.cost_bound_usd:>16.4f}")
        discount = ", Message Batches price (50%)" if self.s.mode == "batch" else ""
        self._say(f"total cost bound ${bound_usd:.4f} (input = ceil(chars/3) tokens at the higher of the input "
                  f"and cache-write rates, output at max_tokens{discount})")
        if self.s.max_spend_usd is not None:
            verdict = "passes" if bound_usd <= self.s.max_spend_usd else "fails"
            self._say(f"pre-flight check against --max-spend-usd ${self.s.max_spend_usd:.4f}: {verdict}")
        if self.skipped_lines:
            self._say(f"warning: {self.skipped_lines} unreadable line(s) in {self.records_path} were ignored")

    # -- execution
    def execute(self, transport: httpx.BaseTransport | None) -> RunReport:
        plans = self.plan()
        bound_usd = sum(plan.cost_bound_usd for plan in plans.values())
        self._print_plan(plans, bound_usd)
        if not self.s.live:
            return self._report(plans)
        if bound_usd > self.s.max_spend_usd:
            raise PreflightRefused(f"pre-flight cost bound ${bound_usd:.4f} exceeds --max-spend-usd "
                                   f"${self.s.max_spend_usd:.4f}; nothing was sent")
        self._write_manifest()
        self.writer = JsonlWriter(self.records_path)
        self.client = AnthropicHTTP(transport, self.s.timeout_s, self.clock)
        try:
            if self.s.mode == "online":
                self._run_online(plans)
            else:
                self.batch_log = JsonlWriter(self.batch_log_path)
                self._run_batch(plans)
        finally:
            self.client.close()
            self.writer.close()
            if self.batch_log is not None:
                self.batch_log.close()
        report = self._report(plans)
        self._print_outcome(report)
        return report

    def _report(self, plans: dict[str, ModelPlan]) -> RunReport:
        return RunReport(self.run_id, self.records_path, self.s.live, plans,
                         {model: dict(counts) for model, counts in self.counts.items()},
                         self.ledger.realized_usd, self.ledger.exhausted, list(self.pending_batches))

    def _print_outcome(self, report: RunReport) -> None:
        for model, counts in report.records_written.items():
            self._say(f"{model}: records written " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        self._say(f"realized cost this invocation ${report.realized_cost_usd:.4f} "
                  f"(limit ${self.s.max_spend_usd:.4f})")
        if report.spend_limit_reached:
            self._say("spend limit reached: unsent examples were recorded as not_attempted; rerun to resume")
        if report.pending_batches:
            self._say(f"batches still processing: {', '.join(report.pending_batches)}; rerun to collect them")

    def _record(self, model: str, item: PromptInput, status: str, outcome: _Outcome, **fields: Any) -> RequestRecord:
        attempt = None
        if status != "not_attempted":
            with self._lock:
                self.attempts[(model, item.example.example_id)] += 1
                attempt = self.attempts[(model, item.example.example_id)]
        return RequestRecord(
            run_id=self.run_id, mode=self.s.mode, model=model, prompt_version=self.prompt.version,
            prompt_sha256=self.prompt.sha256, example_id=item.example.example_id, attempt=attempt,
            stop_reason=outcome.stop_reason, stop_details=outcome.stop_details, usage=outcome.usage,
            cost_usd=outcome.cost_usd, batch_discount_applied=self.s.mode == "batch" and outcome.usage is not None,
            parsed_label=outcome.label if status == "ok" else None, parse_status=status,
            truncated=item.truncated, original_chars=item.original_chars,
            raw_text=outcome.text[:RAW_TEXT_LIMIT] or None, error_type=outcome.error_type, **fields)

    def _write(self, record: RequestRecord) -> None:
        self.writer.write(asdict(record))
        with self._lock:
            self.counts[record.model][record.parse_status] += 1
            written = sum(sum(counts.values()) for counts in self.counts.values())
        if written % PROGRESS_EVERY == 0:
            self._say(f"... {written} records written")

    def _mark_not_attempted(self, model: str, items: Sequence[PromptInput]) -> None:
        for item in items:
            self._write(self._record(model, item, "not_attempted", _Outcome("not_attempted")))

    def _message_outcome(self, model: str, message: Mapping[str, Any], *, batch: bool) -> _Outcome:
        parsed = parse_message(message, self.prompt.schema)
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else None
        cost = self.pricing.cost_usd(model, usage, batch=batch, cache_ttl=self.s.cache_ttl) if usage else 0.0
        return _Outcome(parsed.status, parsed.label, parsed.text, usage, cost,
                        message.get("stop_reason"), message.get("stop_details"))

    # -- online mode
    def _run_online(self, plans: Mapping[str, ModelPlan]) -> None:
        for model in self.s.models:
            plan = plans[model]
            state = _OnlineState()
            if not self.ledger.exhausted:
                with ThreadPoolExecutor(max_workers=self.s.concurrency) as pool:
                    futures = [pool.submit(self._online_example, model, item,
                                           plan.bounds_usd[item.example.example_id], state)
                               for item in plan.pending]
                    try:
                        for future in futures:
                            future.result()
                    except BaseException:
                        state.halt.set()
                        for future in futures:
                            future.cancel()
                        raise
            if self.ledger.exhausted:
                self._mark_not_attempted(model, [item for item in plan.pending
                                                 if item.example.example_id not in state.sent])

    def _online_example(self, model: str, item: PromptInput, bound_usd: float, state: _OnlineState) -> None:
        """Every attempt for one example. An exception halts the model's remaining examples before
        it leaves the worker thread, so no queued example starts after a failure."""
        try:
            body = self.builder.body(model, item.text)
            for attempt in range(1, self.retry.max_attempts + 1):
                if state.halt.is_set() or not self.ledger.reserve(bound_usd):
                    state.halt.set()
                    return
                state.mark_sent(item.example.example_id)
                http = self.client.request("POST", MESSAGES_PATH, json_body=body)
                outcome = self._http_outcome(model, http)
                self.ledger.settle(bound_usd, outcome.cost_usd)
                status = outcome.status
                if status == "retryable":
                    status = "retryable_error" if attempt < self.retry.max_attempts else "missing"
                self._write(self._record(model, item, status, outcome, request_started_utc=http.started_utc,
                                         latency_s=http.latency_s, http_status=http.status,
                                         request_id=http.request_id))
                if status == "aborted":
                    raise RunAborted(f"{model}: {http.describe()}; the run stopped and resumes after the cause is fixed")
                if status != "retryable_error":
                    return
                self.clock.sleep(self.retry.delay_s(attempt, http.retry_after_s, self.rng))
        except BaseException:
            state.halt.set()
            raise

    def _http_outcome(self, model: str, http: HttpAttempt) -> _Outcome:
        if http.retryable:
            return _Outcome("retryable", text=http.text, error_type=http.error_type)
        if http.status in ABORT_STATUS:
            return _Outcome("aborted", text=http.text, error_type=http.error_type)
        if not http.succeeded:
            return _Outcome("error", text=http.text, error_type=http.error_type)
        if not isinstance(http.payload, dict):
            return _Outcome("malformed", text=http.text)
        return self._message_outcome(model, http.payload, batch=False)

    # -- batch mode
    def _run_batch(self, plans: Mapping[str, ModelPlan]) -> None:
        submissions = list(self.recovered)
        for model in self.s.models:
            plan = plans[model]
            submitted: set[str] = set()
            for chunk in [] if self.ledger.exhausted else self._batch_chunks(model, plan.pending):
                accepted, reserved = [], 0.0
                for item, line in chunk:
                    bound = plan.bounds_usd[item.example.example_id]
                    if not self.ledger.reserve(bound):
                        break
                    accepted.append((item, line))
                    reserved += bound
                if accepted:
                    submissions.append(self._submit_batch(model, accepted, reserved))
                    submitted.update(item.example.example_id for item, _ in accepted)
                if self.ledger.exhausted:
                    break
            if self.ledger.exhausted:
                self._mark_not_attempted(model, [item for item in plan.pending
                                                 if item.example.example_id not in submitted])
        self._await_batches(submissions)

    def _batch_chunks(self, model: str, items: Sequence[PromptInput]) -> Iterator[list[tuple[PromptInput, bytes]]]:
        """Serialised batch requests, grouped under the per-batch request-count and size limits."""
        chunk: list[tuple[PromptInput, bytes]] = []
        size = 0
        for item in items:
            request = {"custom_id": encode_custom_id(item.example.example_id),
                       "params": self.builder.body(model, item.text)}
            line = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if chunk and (len(chunk) >= BATCH_MAX_REQUESTS or size + len(line) > BATCH_MAX_BYTES):
                yield chunk
                chunk, size = [], 0
            chunk.append((item, line))
            size += len(line) + 1
        if chunk:
            yield chunk

    def _control(self, method: str, path: str, *, content: bytes | None = None, creates: bool = False) -> HttpAttempt:
        """A batch-endpoint call under the retry policy. A call that creates a batch is resent only
        when the API rejected it before processing (429, 529) or the request never left the client."""
        attempt = 1
        while True:
            http = self.client.request(method, path, content=content)
            if creates:
                retry = http.status in CREATE_RETRY_STATUS or (http.status is None and not http.ambiguous)
            else:
                retry = http.retryable
            if not retry or attempt >= self.retry.max_attempts:
                return http
            self.clock.sleep(self.retry.delay_s(attempt, http.retry_after_s, self.rng))
            attempt += 1

    def _submit_batch(self, model: str, entries: Sequence[tuple[PromptInput, bytes]], reserved_usd: float) -> _Submission:
        content = b'{"requests":[' + b",".join(line for _, line in entries) + b"]}"
        http = self._control("POST", BATCHES_PATH, content=content, creates=True)
        batch = http.payload if http.succeeded and isinstance(http.payload, dict) else {}
        if not isinstance(batch.get("id"), str):
            hint = (" The request may have reached the API: check the Console for an unexpected batch "
                    "before resuming.") if http.status is None or http.status >= 500 else ""
            raise RunAborted(f"{model}: batch creation failed: {http.describe()}.{hint}")
        submission = _Submission(batch["id"], model, tuple(item.example.example_id for item, _ in entries),
                                 http.started_utc, reserved_usd)
        self.batch_log.write({
            "event": "submitted", "run_id": self.run_id, "prompt_version": self.prompt.version, "model": model,
            "batch_id": submission.batch_id, "submitted_utc": submission.submitted_utc,
            "created_at": batch.get("created_at"), "n_requests": len(entries), "cost_bound_usd": reserved_usd,
            "example_ids": list(submission.example_ids),
        })
        self._say(f"{model}: submitted batch {submission.batch_id} with {len(entries)} requests")
        return submission

    def _await_batches(self, submissions: Sequence[_Submission]) -> None:
        """Poll every submitted batch until it ends, collecting each as it does, or until
        --batch-max-wait-s passes (the remaining batches are collected by a later invocation)."""
        waiting = list(submissions)
        deadline = self.clock.monotonic() + self.s.batch_max_wait_s
        while True:
            waiting = [submission for submission in waiting if not self._collect_if_ended(submission)]
            if not waiting:
                return
            if self.clock.monotonic() >= deadline:
                self.pending_batches = [submission.batch_id for submission in waiting]
                return
            self.clock.sleep(self.s.poll_interval_s)

    def _collect_if_ended(self, submission: _Submission) -> bool:
        http = self._control("GET", f"{BATCHES_PATH}/{submission.batch_id}")
        if not http.succeeded or not isinstance(http.payload, dict):
            raise RunAborted(f"batch {submission.batch_id}: status request failed: {http.describe()}")
        if http.payload.get("processing_status") != "ended":
            return False
        self._collect(submission, http.payload)
        return True

    def _collect(self, submission: _Submission, batch: Mapping[str, Any]) -> None:
        http = self._control("GET", f"{BATCHES_PATH}/{submission.batch_id}/results")
        if not http.succeeded:
            raise RunAborted(f"batch {submission.batch_id}: results download failed: {http.describe()}")
        results: dict[str, Mapping[str, Any]] = {}
        for line in http.text.splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("result"), dict):
                results[str(entry.get("custom_id"))] = entry["result"]
        created, ended = batch.get("created_at"), batch.get("ended_at")
        start, end = _parse_utc(created), _parse_utc(ended)
        if start is None or end is None:  # fall back to local submission and observation times
            created, ended = submission.submitted_utc, _iso(self.clock.utc_now())
            start, end = _parse_utc(created), _parse_utc(ended)
        completion_s = (end - start).total_seconds()
        realized = 0.0
        for example_id in submission.example_ids:
            outcome, result_type = self._batch_outcome(submission.model, results.get(encode_custom_id(example_id)))
            realized += outcome.cost_usd
            self._write(self._record(
                submission.model, self.inputs_by_id[example_id], outcome.status, outcome,
                request_started_utc=submission.submitted_utc, batch_id=submission.batch_id,
                batch_created_utc=created, batch_ended_utc=ended, batch_completion_s=completion_s,
                batch_result_type=result_type))
        if submission.reserved_usd is not None:
            self.ledger.settle(submission.reserved_usd, realized)
        self.batch_log.write({
            "event": "collected", "run_id": self.run_id, "batch_id": submission.batch_id,
            "collected_utc": _iso(self.clock.utc_now()), "created_at": created, "ended_at": ended,
            "completion_s": completion_s, "request_counts": batch.get("request_counts"), "n_results": len(results),
        })
        self._say(f"{submission.model}: collected batch {submission.batch_id} ({len(results)} results, "
                  f"completed in {completion_s:.0f} s)")

    def _batch_outcome(self, model: str, result: Mapping[str, Any] | None) -> tuple[_Outcome, str]:
        """Map one batch result; None means the batch returned nothing for the request."""
        if result is None:
            return _Outcome("missing"), "not_returned"
        kind = str(result.get("type"))
        if kind == "succeeded" and isinstance(result.get("message"), dict):
            return self._message_outcome(model, result["message"], batch=True), kind
        if kind == "errored":
            error = result.get("error")
            inner = error.get("error") if isinstance(error, dict) and isinstance(error.get("error"), dict) else error
            error_type = inner.get("type") if isinstance(inner, dict) else None
            status = "error" if error_type in BATCH_EXAMPLE_ERROR_TYPES else "missing"
            return _Outcome(status, text=json.dumps(error, ensure_ascii=False), error_type=error_type), kind
        return _Outcome("missing", text=json.dumps(result, ensure_ascii=False)), kind  # canceled, expired


def run(settings: RunSettings, *, transport: httpx.BaseTransport | None = None, clock: Clock | None = None,
        rng: random.Random | None = None, out: TextIO | None = None) -> RunReport:
    """Plan (dry run) or, with settings.live, execute a run. `transport` replaces httpx's network
    transport (tests pass httpx.MockTransport); `clock` and `rng` make waiting and jitter injectable."""
    return _Run(settings, clock or Clock(), rng or random.Random(), out or sys.stdout).execute(transport)


# --- Summary ------------------------------------------------------------------------------------
SUMMARY_DEFINITIONS = {
    "final_status": "per example: its first terminal record (ok, malformed, refusal, max_tokens, error); "
                    "otherwise not_attempted if its last record is a spend-limit marker, else missing",
    "accuracy_all_eligible": "correct / n_eligible; every example whose final status is not ok "
                             "(malformed, refusal, max_tokens, error, missing, not_attempted) counts as incorrect",
    "accuracy_parsed_ok": "correct / n_ok, over the examples whose final status is ok",
    "error_rate_by_true_class": "per true class, (examples of that class not classified correctly) / (examples "
                                "of that class), over all eligible examples (non-ok counts as an error) and over "
                                "the parsed-ok subset",
    "tokens": "usage summed over every attempt record of the eligible examples; per_example_mean divides by "
              "n_examples_with_usage",
    "cost_usd": "sum of record cost_usd (pricing.json applied to the reported usage, batch discount included)",
    "latency_s": "online mode only: wall time of each HTTP request that returned HTTP 200 (linear-interpolation "
                 "quantiles)",
    "batches": "batch mode only: per batch, ended_at minus created_at as reported by the API; span_s runs from "
               "the earliest creation to the latest end. Batch completion time is not a per-request latency",
}
PER_EXAMPLE_COLUMNS = ("run_id", "model", "mode", "prompt_version", "example_id", "label", "final_status",
                       "predicted_label", "correct", "n_attempts", *TOKEN_FIELDS, "cost_usd", "latency_s",
                       "truncated", "original_chars", "batch_id")


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {"value": numerator / denominator if denominator else None, "numerator": numerator,
            "denominator": denominator}


def _latency_summary(latencies: Sequence[float]) -> dict[str, Any] | None:
    if not latencies:
        return None
    median, p90, p99 = np.percentile(np.asarray(latencies, dtype=float), [50, 90, 99])
    return {"n_requests": len(latencies), "median": float(median), "p90": float(p90), "p99": float(p99)}


def _batch_summary(batches: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    per_batch = sorted(batches.values(), key=lambda b: str(b["created_utc"]))
    starts = [_parse_utc(b["created_utc"]) for b in per_batch]
    ends = [_parse_utc(b["ended_utc"]) for b in per_batch]
    span = (max(ends) - min(starts)).total_seconds() if per_batch and None not in starts + ends else None
    return {"n_batches": len(per_batch), "per_batch": per_batch, "span_s": span}


def _summarize_group(key: tuple[Any, ...], records: Sequence[Mapping[str, Any]],
                     examples: Sequence[Example]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id, model, mode, prompt_version = key
    labels = {example.example_id: example.label for example in examples}
    by_example: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("example_id") in labels:
            by_example[record["example_id"]].append(record)
    statuses = Counter({status: 0 for status in FINAL_STATUSES})
    n_class, n_class_ok, n_class_correct, tokens = Counter(), Counter(), Counter(), Counter()
    cost, with_usage, duplicates, latencies, batches, rows = 0.0, 0, 0, [], {}, []
    for example in examples:
        recs = by_example.get(example.example_id, [])
        terminal = [r for r in recs if r.get("parse_status") in TERMINAL_STATUSES]
        duplicates += max(len(terminal) - 1, 0)
        final = terminal[0] if terminal else (recs[-1] if recs else {})
        if terminal:
            status = str(final["parse_status"])
        else:
            status = "not_attempted" if final.get("parse_status") == "not_attempted" else "missing"
        statuses[status] += 1
        predicted = final.get("parsed_label") if status == "ok" else None
        correct = status == "ok" and predicted == example.label
        true_class = INT_TO_LABEL[example.label]
        n_class[true_class] += 1
        n_class_ok[true_class] += status == "ok"
        n_class_correct[true_class] += correct
        example_tokens, example_cost = Counter(), 0.0
        for record in recs:
            usage = record.get("usage")
            if isinstance(usage, dict):
                example_tokens.update({name: _usage_count(usage, name) for name in TOKEN_FIELDS})
            example_cost += float(record.get("cost_usd") or 0.0)
            if mode == "online" and record.get("http_status") == 200 and isinstance(record.get("latency_s"), (int, float)):
                latencies.append(float(record["latency_s"]))
            if record.get("batch_id"):
                info = batches.setdefault(record["batch_id"], {
                    "batch_id": record["batch_id"], "created_utc": record.get("batch_created_utc"),
                    "ended_utc": record.get("batch_ended_utc"), "completion_s": record.get("batch_completion_s"),
                    "n_records": 0})
                info["n_records"] += 1
        with_usage += any(isinstance(record.get("usage"), dict) for record in recs)
        tokens.update(example_tokens)
        cost += example_cost
        rows.append({
            "run_id": run_id, "model": model, "mode": mode, "prompt_version": prompt_version,
            "example_id": example.example_id, "label": example.label, "final_status": status,
            "predicted_label": predicted, "correct": int(correct),
            "n_attempts": sum(r.get("parse_status") != "not_attempted" for r in recs),
            **{name: example_tokens[name] if recs else None for name in TOKEN_FIELDS},
            "cost_usd": example_cost if recs else None, "latency_s": final.get("latency_s"),
            "truncated": final.get("truncated"), "original_chars": final.get("original_chars"),
            "batch_id": final.get("batch_id"),
        })
    n_correct = sum(n_class_correct.values())
    summary = {
        "run_id": run_id, "model": model, "mode": mode, "prompt_version": prompt_version,
        "prompt_sha256": sorted({str(record.get("prompt_sha256")) for record in records}),
        "n_eligible": len(examples),
        "n_by_final_status": dict(statuses),
        "accuracy_all_eligible": _ratio(n_correct, len(examples)),
        "accuracy_parsed_ok": _ratio(n_correct, statuses["ok"]),
        "error_rate_by_true_class": {
            name: {"all_eligible": _ratio(n_class[name] - n_class_correct[name], n_class[name]),
                   "parsed_ok": _ratio(n_class_ok[name] - n_class_correct[name], n_class_ok[name])}
            for name in LABEL_TO_INT},
        "tokens": {"total": {name: tokens[name] for name in TOKEN_FIELDS},
                   "per_example_mean": {name: tokens[name] / with_usage for name in TOKEN_FIELDS} if with_usage else None,
                   "n_examples_with_usage": with_usage},
        "cost_usd": {"total": cost, "per_example_mean": cost / with_usage if with_usage else None},
        "n_records": len(records),
        "n_records_outside_eligible_set": len(records) - sum(len(v) for v in by_example.values()),
        "n_duplicate_terminal_records": duplicates,
    }
    if mode == "online":
        summary["latency_s"] = _latency_summary(latencies)
    if mode == "batch":
        summary["batches"] = _batch_summary(batches)
    return summary, rows


def summarize(records_paths: Sequence[Path], examples_path: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Write summary.json and per_example.csv for every (run, model, mode, prompt version) that has
    records; the eligible set is the examples file, checked against each run's manifest."""
    examples = sorted(load_examples(examples_path), key=lambda e: e.example_id)
    eval_sha256 = example_set_sha256(examples)
    records: list[dict[str, Any]] = []
    unreadable = 0
    for path in records_paths:
        if not path.is_file():
            raise InputError(f"{path}: file not found")
        found, skipped = read_jsonl(path)
        records += found
        unreadable += skipped
        manifest_path = path.parent / MANIFEST_NAME
        if manifest_path.is_file():
            stored = json.loads(manifest_path.read_text(encoding="utf-8")).get("config", {}).get("eval_set_sha256")
            if stored not in (None, eval_sha256):
                raise InputError(f"{examples_path} is not the evaluation set of the run in {path.parent} "
                                 "(eligible-set sha256 differs)")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record.get("run_id"), record.get("model"), record.get("mode"), record.get("prompt_version"))].append(record)
    summaries, rows = [], []
    for key in sorted(groups, key=lambda k: tuple(str(part) for part in k)):
        summary, group_rows = _summarize_group(key, groups[key], examples)
        summaries.append(summary)
        rows += group_rows
    document = {
        "generated_utc": _iso(_utc_now()),
        "examples": {"path": str(examples_path), "n": len(examples), "eligible_set_sha256": eval_sha256},
        "records": [str(path) for path in records_paths],
        "n_unreadable_record_lines": unreadable,
        "definitions": SUMMARY_DEFINITIONS,
        "groups": summaries,
    }
    out_dir = out_dir or records_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "per_example.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_EXAMPLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return document


# --- Command line -------------------------------------------------------------------------------
def _model_list(value: str) -> tuple[str, ...]:
    models = tuple(part.strip() for part in value.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("expected a comma-separated list of model ids")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m study.llm.runner",
                                     description="Claude zero/few-shot sentiment baseline runner.")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("run", help="dry run by default; --live sends requests")
    plan.add_argument("--examples", type=Path, required=True, help="evaluation CSV: example_id,text,label")
    plan.add_argument("--exemplars", type=Path, help="few-shot CSV (ids must start with train/)")
    plan.add_argument("--models", type=_model_list, default=DEFAULT_MODELS,
                      help=f"comma-separated model ids (default {','.join(DEFAULT_MODELS)})")
    plan.add_argument("--mode", choices=("online", "batch"), default="online")
    plan.add_argument("--prompt-version", default="v1")
    plan.add_argument("--max-review-chars", type=int, help="truncate longer reviews (default: no truncation)")
    plan.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    plan.add_argument("--cache-ttl", choices=("5m", "1h", "off"), default="5m")
    plan.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    plan.add_argument("--concurrency", type=int, default=1, help="online mode: concurrent requests")
    plan.add_argument("--run-id", help="default: <prompt version>-<12 hex digits of the configuration hash>")
    plan.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    plan.add_argument("--records", type=Path, help="default: <results-dir>/<run-id>/requests.jsonl")
    plan.add_argument("--live", action="store_true", help="send requests (requires --max-spend-usd)")
    plan.add_argument("--max-spend-usd", type=float, help="spend limit for this invocation")
    plan.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    plan.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    plan.add_argument("--batch-max-wait-s", type=float, default=DEFAULT_BATCH_MAX_WAIT_S)
    report = commands.add_parser("summarize", help="write summary.json and per_example.csv")
    report.add_argument("--records", type=Path, nargs="+", required=True)
    report.add_argument("--examples", type=Path, required=True)
    report.add_argument("--out-dir", type=Path, help="default: the directory of the first records file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and args.live and args.max_spend_usd is None:
        parser.error("--live requires --max-spend-usd")
    try:
        if args.command == "run":
            run(RunSettings(
                examples_path=args.examples, exemplars_path=args.exemplars, models=args.models, mode=args.mode,
                prompt_version=args.prompt_version, max_review_chars=args.max_review_chars,
                max_tokens=args.max_tokens, cache_ttl=args.cache_ttl, max_attempts=args.max_attempts,
                concurrency=args.concurrency, run_id=args.run_id, results_dir=args.results_dir,
                records_path=args.records, live=args.live, max_spend_usd=args.max_spend_usd,
                timeout_s=args.timeout_s, poll_interval_s=args.poll_interval_s,
                batch_max_wait_s=args.batch_max_wait_s))
        else:
            document = summarize(args.records, args.examples, args.out_dir)
            for group in document["groups"]:
                accuracy = group["accuracy_all_eligible"]
                print(f"{group['model']} ({group['mode']}, {group['run_id']}): accuracy {accuracy['numerator']}/"
                      f"{accuracy['denominator']} on all eligible examples, statuses {group['n_by_final_status']}")
            print(f"wrote {(args.out_dir or args.records[0].parent) / 'summary.json'} and per_example.csv")
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
