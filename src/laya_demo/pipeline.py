"""Run the whole demo: Laya answers, route by confidence, escalate to Gemini, finalize."""

import random
import time
from collections.abc import Callable
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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def check_signal(signal: str) -> None:
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}; use one of {SIGNALS}")


def load_system1(cfg: Config, labels: list[str], system1=None, log=print):
    """Load Laya for these labels, or check that the one passed in fits them, then warm it up."""
    from .system1 import LayaSystem1

    if system1 is None:
        log(f"Loading Laya checkpoint '{cfg.laya_model}' (downloads on first use)")
        system1 = LayaSystem1(labels, model=cfg.laya_model, device=cfg.device)
    elif system1.labels != labels or system1.model != cfg.laya_model:
        raise ValueError("the system1 passed in was built for different labels or a different checkpoint")
    system1.warm_up()
    return system1


def run_laya(items, laya, on_item: Callable[[Record], None] | None = None) -> list[Record]:
    """Answer every item with Laya, one call each, calling `on_item` after each answer."""
    records = []
    for item in items:
        a = laya.answer(item.text)
        record = Record(item.idx, item.text, item.label, a.answer, a.confidence, a.top_prob, a.latency_ms)
        records.append(record)
        if on_item is not None:
            on_item(record)
    return records


def n_requests(n_items: int, batch_size: int) -> int:
    return -(-n_items // batch_size)


def gemini_status(model: str, system2, answers: list[System2Answer], batch_size: int) -> str:
    """One line on what System 2 did: items, requests, errors, and which models actually answered."""
    errors = [a.error for a in answers if a.error]
    status = f"{model}, {len(answers)} items in {n_requests(len(answers), batch_size)} requests, {len(errors)} errors"
    if set(system2.answered_by) - {model}:
        used = ", ".join(f"{m} x{n}" for m, n in system2.answered_by.items())
        status += f"; overloaded, so requests fell back: {used}"
    if errors:
        status += f" (first: {errors[0]})"
    return status


def gemini_unavailable(use_gemini: bool, key: str | None) -> str | None:
    """The status line when Gemini will not be called, or None when it will."""
    if not use_gemini:
        return "disabled with --no-gemini; escalated items keep Laya's answer"
    if key is None:
        return "skipped: no gemini_api_key in .env; escalated items keep Laya's answer"
    return None


@dataclass
class GeminiOutcome:
    model: str
    status: str
    baseline_idx: list[int] = field(default_factory=list)


def run_gemini(
    records: list[Record],
    labels: list[str],
    cfg: Config,
    log=print,
    system2=None,
    on_batch_start=None,
    on_batch_done=None,
) -> GeminiOutcome:
    """Send the escalated items, then a baseline sample, to Gemini within the cap, and store its answers.

    Records must already be routed. Pass a built `GeminiSystem2` as `system2` to skip the key lookup.
    """
    from .system2 import GeminiSystem2, default_model, load_api_key

    model = cfg.gemini_model or default_model()
    if system2 is None:
        key = load_api_key() if cfg.use_gemini else None
        unavailable = gemini_unavailable(cfg.use_gemini, key)
        if unavailable:
            return GeminiOutcome(model, unavailable)
        system2 = GeminiSystem2(labels, key, model=model)
    model = system2.model

    calls, baseline_idx = plan_gemini_calls(records, cfg.gemini_baseline, cfg.max_gemini_calls, cfg.seed)
    n_escalated = sum(r.action == ESCALATE for r in records)
    log(
        f"Calling {model} on {len(calls)} items in {n_requests(len(calls), cfg.gemini_batch_size)} batched requests "
        f"({n_escalated} escalated, cap {cfg.max_gemini_calls})"
    )
    by_idx = {r.idx: r for r in records}
    answers = system2.answer_many(
        [by_idx[i].text for i in calls],
        batch_size=cfg.gemini_batch_size,
        on_batch_start=on_batch_start,
        on_batch_done=on_batch_done,
    )
    apply_gemini(records, dict(zip(calls, answers, strict=True)))
    status = gemini_status(model, system2, answers, cfg.gemini_batch_size)
    skipped = n_escalated - sum(1 for i in calls if by_idx[i].action == ESCALATE)
    if skipped:
        status += f"; {skipped} escalated items over the cap keep Laya's answer"
    return GeminiOutcome(model, status, baseline_idx)


def build_meta(cfg: Config, started: str, duration_s: float, device: str, gemini: GeminiOutcome, n: int) -> dict:
    meta = {
        "started_at": started,
        "duration_s": round(duration_s, 1),
        "dataset": "legacy-datasets/banking77 test",
        "n": n,
        "seed": cfg.seed,
        "thresholds": list(cfg.thresholds),
        "signal": cfg.signal,
        "laya_model": cfg.laya_model,
        "device": device,
        "gemini_model": gemini.model,
        "gemini_status": gemini.status,
        "max_gemini_calls": cfg.max_gemini_calls,
        **cfg.extra_meta,
    }
    meta["laya_version"] = version("laya")
    return meta


def run_pipeline(cfg: Config, log=print, system1=None) -> RunResult:
    """Run the demo end to end. Pass an already loaded `LayaSystem1` as `system1` to reuse it."""
    check_signal(cfg.signal)
    tiers = make_tiers(*cfg.thresholds)
    started = now_iso()
    t0 = time.perf_counter()

    log(f"Loading Banking77 sample (n={cfg.n}, seed={cfg.seed})")
    items, labels = load_banking77(cfg.n, cfg.seed)
    laya = load_system1(cfg, labels, system1, log)
    log(f"Laya on {laya.device}; answering {len(items)} items")
    records = run_laya(items, laya)
    route_records(records, tiers, cfg.signal)

    gemini = run_gemini(records, labels, cfg, log)
    log(f"System 2: {gemini.status}")
    finalize(records)

    meta = build_meta(cfg, started, time.perf_counter() - t0, laya.device, gemini, len(records))
    return RunResult(meta, labels, records, gemini.baseline_idx)
