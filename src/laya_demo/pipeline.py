"""Run the whole demo: Laya answers, route by confidence, escalate to Gemini, finalize."""

import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version

from .data import load_banking77
from .records import Record, RunResult
from .routing import ACCEPT, ESCALATE, PAPER_THRESHOLDS, assign_tier, make_tiers
from .system2 import System2Answer

SIGNALS = ("confidence", "top_prob")


@dataclass
class Config:
    n: int = 150
    seed: int = 0
    thresholds: tuple[float, float, float] = PAPER_THRESHOLDS
    signal: str = "confidence"  # which Laya number to route on: its `confidence` field or the top probability
    laya_model: str = "english"
    device: str | None = None
    use_gemini: bool = True
    gemini_model: str | None = None
    max_gemini_calls: int = 60
    gemini_baseline: int = 30  # random items Gemini also answers, to measure Gemini alone
    gemini_batch_size: int = 30  # items per Gemini request; the free tier allows 5 requests a minute
    extra_meta: dict = field(default_factory=dict)


def signal_value(record: Record, signal: str) -> float:
    if signal == "confidence":
        return record.laya_confidence
    if signal == "top_prob":
        return record.laya_top_prob
    raise ValueError(f"unknown signal {signal!r}; use one of {SIGNALS}")


def route_records(records: list[Record], tiers, signal: str) -> None:
    """Set each record's tier and action from its confidence signal."""
    for r in records:
        tier = assign_tier(signal_value(r, signal), tiers)
        r.tier, r.action = tier.name, tier.action


def plan_gemini_calls(records: list[Record], baseline_n: int, cap: int, seed: int) -> tuple[list[int], list[int]]:
    """Choose which items Gemini answers, within the cap.

    Escalated items come first because the routed system needs them. The rest of the budget
    goes to a seeded uniform sample, so the Gemini alone number is not biased toward the hard
    items that Laya escalated. Returns (indices to call, in order; baseline indices kept).
    """
    escalated = [r.idx for r in records if r.action == ESCALATE]
    rng = random.Random(seed)
    baseline = sorted(rng.sample([r.idx for r in records], min(baseline_n, len(records))))
    order = escalated + [i for i in baseline if i not in set(escalated)]
    calls = order[: max(0, cap)]
    called = set(calls)
    return calls, [i for i in baseline if i in called]


def finalize(records: list[Record]) -> None:
    """Pick each item's final answer from its tier's action and what Gemini returned."""
    for r in records:
        if r.action == ACCEPT:
            r.final_answer, r.final_source = r.laya_answer, "laya"
        elif r.action == ESCALATE and r.gemini_answer is not None:
            r.final_answer, r.final_source = r.gemini_answer, "gemini"
        elif r.action == ESCALATE:
            # No key, the cap ran out, or the call failed: keep Laya's answer and say so.
            r.final_answer, r.final_source = r.laya_answer, "laya_fallback"
        else:
            r.final_answer, r.final_source = None, "human"


def apply_gemini(records: list[Record], answers: dict[int, System2Answer]) -> None:
    by_idx = {r.idx: r for r in records}
    for idx, ans in answers.items():
        r = by_idx[idx]
        r.gemini_answer, r.gemini_latency_ms, r.gemini_error = ans.answer, ans.latency_ms, ans.error


def run_pipeline(cfg: Config, log=print, system1=None) -> RunResult:
    """Run the demo end to end. Pass an already loaded `LayaSystem1` as `system1` to reuse it."""
    if cfg.signal not in SIGNALS:
        raise ValueError(f"unknown signal {cfg.signal!r}; use one of {SIGNALS}")
    tiers = make_tiers(*cfg.thresholds)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.perf_counter()

    log(f"Loading Banking77 sample (n={cfg.n}, seed={cfg.seed})")
    items, labels = load_banking77(cfg.n, cfg.seed)

    from .system1 import LayaSystem1

    if system1 is None:
        log(f"Loading Laya checkpoint '{cfg.laya_model}' (downloads on first use)")
        system1 = LayaSystem1(labels, model=cfg.laya_model, device=cfg.device)
    elif system1.labels != labels or system1.model != cfg.laya_model:
        raise ValueError("the system1 passed in was built for different labels or a different checkpoint")
    laya = system1
    laya.warm_up()
    log(f"Laya on {laya.device}; answering {len(items)} items")
    records = []
    for item in items:
        a = laya.answer(item.text)
        records.append(Record(item.idx, item.text, item.label, a.answer, a.confidence, a.top_prob, a.latency_ms))
    route_records(records, tiers, cfg.signal)

    from .system2 import GeminiSystem2, default_model, load_api_key

    gemini_model = cfg.gemini_model or default_model()
    baseline_idx: list[int] = []
    key = load_api_key() if cfg.use_gemini else None
    if not cfg.use_gemini:
        status = "disabled with --no-gemini; escalated items keep Laya's answer"
    elif key is None:
        status = "skipped: no gemini_api_key in .env; escalated items keep Laya's answer"
    else:
        calls, baseline_idx = plan_gemini_calls(records, cfg.gemini_baseline, cfg.max_gemini_calls, cfg.seed)
        n_escalated = sum(r.action == ESCALATE for r in records)
        n_requests = -(-len(calls) // cfg.gemini_batch_size)
        log(
            f"Calling {gemini_model} on {len(calls)} items in {n_requests} batched requests "
            f"({n_escalated} escalated, cap {cfg.max_gemini_calls})"
        )
        system2 = GeminiSystem2(labels, key, model=gemini_model)
        by_idx = {r.idx: r for r in records}
        answers = system2.answer_many([by_idx[i].text for i in calls], batch_size=cfg.gemini_batch_size)
        apply_gemini(records, dict(zip(calls, answers, strict=True)))
        errors = [a.error for a in answers if a.error]
        status = f"{gemini_model}, {len(calls)} items in {n_requests} requests, {len(errors)} errors"
        if set(system2.answered_by) - {gemini_model}:
            used = ", ".join(f"{m} x{n}" for m, n in system2.answered_by.items())
            status += f"; overloaded, so requests fell back: {used}"
        if errors:
            status += f" (first: {errors[0]})"
        skipped = n_escalated - sum(1 for i in calls if by_idx[i].action == ESCALATE)
        if skipped:
            status += f"; {skipped} escalated items over the cap keep Laya's answer"
    log(f"System 2: {status}")
    finalize(records)

    meta = {
        "started_at": started,
        "duration_s": round(time.perf_counter() - t0, 1),
        "dataset": "legacy-datasets/banking77 test",
        "n": len(records),
        "seed": cfg.seed,
        "thresholds": list(cfg.thresholds),
        "signal": cfg.signal,
        "laya_model": cfg.laya_model,
        "device": laya.device,
        "gemini_model": gemini_model,
        "gemini_status": status,
        "max_gemini_calls": cfg.max_gemini_calls,
        **cfg.extra_meta,
    }
    meta["laya_version"] = version("laya")
    return RunResult(meta, labels, records, baseline_idx)
