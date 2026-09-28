"""Gemini as System 2, through `google-genai`.

The key is read from `.env` as `gemini_api_key` with python-dotenv, or from the environment
variable of the same name. It is passed to the client and never printed or stored.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# The model the installed google-genai release uses in its own tests. Override with
# --gemini-model or the GEMINI_MODEL environment variable if your key has a different one.
DEFAULT_MODEL = "gemini-3.5-flash"
KEY_NAME = "gemini_api_key"

SYSTEM_PROMPT = (
    "You classify customer messages sent to a bank's support channel. "
    "Answer with exactly one intent label from the allowed list."
)


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


class GeminiSystem2:
    """Classify with Gemini, constrained to the label list by an enum response schema."""

    def __init__(self, labels: list[str], api_key: str, model: str | None = None, timeout_s: float = 30.0):
        from google import genai
        from google.genai import types

        self.labels = labels
        self.model = model or default_model()
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="text/x.enum",
            response_schema={"type": "STRING", "enum": labels},
        )

    def answer(self, text: str) -> System2Answer:
        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(model=self.model, contents=text, config=self._config)
            latency_ms = (time.perf_counter() - start) * 1000
            label = (response.text or "").strip()
        # Deliberately broad: during a live demo one failed call (network, quota, a model name the
        # key cannot use, an SDK error type not seen before) must be recorded, not end the run.
        except Exception as exc:  # noqa: BLE001
            return System2Answer(None, (time.perf_counter() - start) * 1000, f"{type(exc).__name__}: {exc}"[:300])
        if label not in self.labels:
            return System2Answer(None, latency_ms, f"answer not in label list: {label[:80]!r}")
        return System2Answer(label, latency_ms)

    def answer_many(self, texts: list[str], workers: int = 8) -> list[System2Answer]:
        """Answer several texts concurrently. Each latency is still measured per call."""
        if not texts:
            return []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.answer, texts))
