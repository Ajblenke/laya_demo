"""Laya as System 1: one typed `choice` question over the 77 Banking77 intents.

Verified against the installed laya 0.3.10 source:

* `Router().predict(state, questions, model=...)` routes to a checkpoint and returns
  `{"model", "answers", "usage", "routing"}`. `model` accepts `english`, `multilingual`,
  `typed-decisions` and aliases. `typed-decisions` is fine tuned on four synthetic workflows,
  so this demo uses `english`, the checkpoint the Router picks for English text anyway.
* A `choice` answer carries `choice`, `probabilities` (temperature scaled softmax),
  `confidence` and `action.act_probability`.
* `confidence` is not the top probability. It is `1 - H(p) / log(k)`, one minus normalized
  entropy (`laya.common.confidence_from_probs`). The demo records both.
"""

import time
from dataclasses import dataclass

from .data import readable

QUESTION_ID = "intent"
INSTRUCTIONS = "Which banking customer intent does this message express?"


@dataclass(frozen=True)
class System1Answer:
    answer: str
    confidence: float
    top_prob: float
    latency_ms: float


def build_question(labels: list[str]) -> dict:
    """One `choice` question whose options are the readable form of every label."""
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": [readable(label) for label in labels],
        }
    }


def parse_answer(payload: dict, labels: list[str]) -> tuple[str, float, float]:
    """Map a Laya `choice` answer back to a dataset label, its confidence, and its top probability."""
    by_readable = {readable(label): label for label in labels}
    choice = payload["choice"]
    if choice not in by_readable:
        raise ValueError(f"Laya returned an option that is not a known label: {choice!r}")
    top_prob = max(payload["probabilities"].values())
    return by_readable[choice], float(payload["confidence"]), float(top_prob)


class LayaSystem1:
    """Load one Laya checkpoint and answer items one at a time, timing each call."""

    def __init__(self, labels: list[str], model: str = "english", device: str | None = None):
        from laya import Router

        self.labels = labels
        self.model = model
        self.question = build_question(labels)
        self.router = Router(device=device)
        self.agent = self.router.load(model)  # downloads the checkpoint on first use
        self.device = str(self.agent.device)

    def warm_up(self) -> None:
        """Run one untimed call so CUDA setup does not land on the first item's latency."""
        self.router.predict({"message": "warm up"}, self.question, model=self.model)

    def answer(self, text: str) -> System1Answer:
        start = time.perf_counter()
        result = self.router.predict({"message": text}, self.question, model=self.model)
        latency_ms = (time.perf_counter() - start) * 1000
        label, confidence, top_prob = parse_answer(result["answers"][QUESTION_ID], self.labels)
        return System1Answer(label, confidence, top_prob, latency_ms)
