"""Gemini as System 2, through `google-genai`.

The key is read from `.env` as `gemini_api_key` with python-dotenv, or from the environment
variable of the same name. It is passed to the client and never printed or stored.

The free tier allows 5 requests per minute per model, so messages are classified in batches,
one request per batch, and each item's latency is its batch's latency divided by the batch size.
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

# The model the installed google-genai release uses in its own tests. Override with
# --gemini-model or the GEMINI_MODEL environment variable if your key has a different one.
DEFAULT_MODEL = "gemini-3.5-flash"
# Tried in order when the chosen model stays overloaded (503) or out of quota (429) after its
# retries. Free tier demand moves between models hour to hour, so a live demo needs more than one.
FALLBACK_MODELS = ("gemini-3.6-flash", "gemini-3.1-flash-lite")
KEY_NAME = "gemini_api_key"

SYSTEM_PROMPT = (
    "You classify customer messages sent to a bank's support channel. "
    "Each message is numbered. For every message, return its id and exactly one intent label "
    "from the allowed list."
)
DEFAULT_BATCH_SIZE = 30
RATE_LIMIT_RETRIES = 2
MAX_WAIT_S = 20.0  # never stall a live demo longer than this on one retry
_RETRY_DELAY = re.compile(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s")


@dataclass(frozen=True)
class System2Answer:
    answer: str | None
    latency_ms: float
    error: str | None = None


def load_api_key() -> str | None:
    """Return the Gemini key from `.env` (searched upward from the working directory) or the environment."""
    from dotenv import dotenv_values, find_dotenv

    path = find_dotenv(usecwd=True)
    value = dotenv_values(path).get(KEY_NAME) if path else None
    value = value or os.environ.get(KEY_NAME)
    return value.strip() if value and value.strip() else None


def default_model() -> str:
    return os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def format_batch(texts: list[str]) -> str:
    return "\n".join(f"[{i}] {' '.join(t.split())}" for i, t in enumerate(texts))


def parse_batch(reply: str, n: int, labels: list[str]) -> list[tuple[str | None, str | None]]:
    """Map a batch reply back to (answer, error) per input position, in order."""
    try:
        rows = json.loads(reply)
    except json.JSONDecodeError:
        return [(None, "batch reply was not valid JSON")] * n
    if not isinstance(rows, list):
        return [(None, "batch reply was not a JSON list")] * n
    by_id: dict[int, str] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), int) and isinstance(row.get("label"), str):
            by_id.setdefault(row["id"], row["label"].strip())
    out: list[tuple[str | None, str | None]] = []
    for i in range(n):
        label = by_id.get(i)
        if label is None:
            out.append((None, "missing from batch reply"))
        elif label not in labels:
            out.append((None, f"answer not in label list: {label[:80]!r}"))
        else:
            out.append((label, None))
    return out


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def retry_delay_s(exc: Exception) -> float | None:
    """How long to wait before retrying a transient error, or None when retrying will not help.

    A 429 rate limit waits the server's suggested delay; a 503 "high demand" waits a few seconds.
    """
    text = str(exc)
    if is_rate_limit(exc):
        match = _RETRY_DELAY.search(text)
        return float(match.group(1)) if match else 15.0
    if "503" in text or "UNAVAILABLE" in text:
        return 4.0
    return None


class GeminiSystem2:
    """Classify with Gemini in batches, constrained to the label list by a JSON response schema."""

    def __init__(self, labels: list[str], api_key: str, model: str | None = None, timeout_s: float = 30.0):
        from google import genai
        from google.genai import types

        self.labels = labels
        self.model = model or default_model()
        self.fallbacks = [m for m in FALLBACK_MODELS if m != self.model]
        self.answered_by: Counter[str] = Counter()
        self._exhausted: set[str] = set()
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"id": {"type": "INTEGER"}, "label": {"type": "STRING", "enum": labels}},
                    "required": ["id", "label"],
                },
            },
        )
        self._sleep = time.sleep

    def _generate_with(self, model: str, contents: str, last: bool) -> str:
        """One request to one model, retried after a wait when the error is transient.

        Quota is per model, so a 429 with a fallback left moves on at once instead of waiting.
        """
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = self._client.models.generate_content(model=model, contents=contents, config=self._config)
                return response.text or ""
            except Exception as exc:
                delay = retry_delay_s(exc)
                if delay is None or attempt == RATE_LIMIT_RETRIES or (is_rate_limit(exc) and not last):
                    raise
                self._sleep(min(delay, MAX_WAIT_S) + 1)
        raise AssertionError("unreachable")

    def _generate(self, contents: str) -> str:
        """Try the chosen model, then each fallback, until one answers; re-raise the last failure.

        A model that stays overloaded is skipped for the rest of the run, so later batches do not
        wait through its retries again.
        """
        models = [m for m in (self.model, *self.fallbacks) if m not in self._exhausted]
        for i, model in enumerate(models):
            try:
                reply = self._generate_with(model, contents, last=i == len(models) - 1)
            except Exception as exc:
                if i == len(models) - 1 or retry_delay_s(exc) is None:
                    raise
                self._exhausted.add(model)
                continue
            self.answered_by[model] += 1
            return reply
        raise AssertionError("unreachable")

    def answer_batch(self, texts: list[str]) -> list[System2Answer]:
        """Classify several texts in one request. Every item gets the batch latency divided evenly."""
        start = time.perf_counter()
        try:
            reply = self._generate(format_batch(texts))
        # Deliberately broad: during a live demo a failed call (network, quota, a model name the
        # key cannot use, an SDK error type not seen before) must be recorded, not end the run.
        except Exception as exc:  # noqa: BLE001
            per_item = (time.perf_counter() - start) * 1000 / len(texts)
            return [System2Answer(None, per_item, f"{type(exc).__name__}: {exc}"[:300]) for _ in texts]
        per_item = (time.perf_counter() - start) * 1000 / len(texts)
        return [System2Answer(a, per_item, e) for a, e in parse_batch(reply, len(texts), self.labels)]

    def answer(self, text: str) -> System2Answer:
        return self.answer_batch([text])[0]

    def answer_many(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[System2Answer]:
        """Answer texts in sequential batches of `batch_size`, one request each, in input order."""
        out: list[System2Answer] = []
        for start in range(0, len(texts), batch_size):
            out.extend(self.answer_batch(texts[start : start + batch_size]))
        return out
