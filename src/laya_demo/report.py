"""Per tier and overall summaries of a run, as plain rows and as text tables."""

from statistics import fmean

from .calibration import expected_calibration_error
from .records import Record, RunResult
from .routing import ESCALATE, HUMAN, Tier, tier_bounds


def _accuracy(flags: list[bool]) -> float | None:
    return fmean(flags) if flags else None


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def tier_table(records: list[Record], tiers: tuple[Tier, ...]) -> list[dict]:
    """One row per tier: how many items landed there and how the routed system did on them.

    `laya_acc` is what Laya alone would have scored in that tier. `final_acc` is the routed
    answer's accuracy, left empty for the human tier because no model answers it.
    """
    bounds = tier_bounds(tiers)
    total = len(records)
    rows = []
    for tier in tiers:
        rs = [r for r in records if r.tier == tier.name]
        finals = [r.final_correct for r in rs if r.final_correct is not None]
        rows.append(
            {
                "tier": tier.name,
                "range": bounds[tier.name],
                "paper": tier.paper_action,
                "demo": {"accept": "accept Laya", ESCALATE: "Gemini", HUMAN: "flag for human"}[tier.action],
                "count": len(rs),
                "share": len(rs) / total if total else 0.0,
                "laya_acc": _accuracy([r.laya_correct for r in rs]),
                "final_acc": None if tier.action == HUMAN else _accuracy(finals),
                "gemini_calls": sum(r.gemini_answer is not None or r.gemini_error is not None for r in rs)
                if tier.action == ESCALATE
                else 0,
                "mean_latency_ms": _mean([r.routed_latency_ms for r in rs]),
            }
        )
    return rows


def tradeoff_table(run: RunResult) -> list[dict]:
    """Laya alone, Gemini alone, and the routed system, side by side."""
    records = run.records
    by_idx = {r.idx: r for r in records}
    baseline = [by_idx[i] for i in run.gemini_baseline_idx if by_idx[i].gemini_answer is not None]
    automated = [r for r in records if r.final_correct is not None]
    return [
        {
            "system": "Laya alone",
            "n": len(records),
            "accuracy": _accuracy([r.laya_correct for r in records]),
            "coverage": 1.0 if records else 0.0,
            "mean_latency_ms": _mean([r.laya_latency_ms for r in records]),
        },
        {
            "system": "Gemini alone",
            "n": len(baseline),
            "accuracy": _accuracy([r.gemini_correct for r in baseline]),
            "coverage": 1.0 if baseline else 0.0,
            "mean_latency_ms": _mean([r.gemini_latency_ms for r in baseline]),
        },
        {
            "system": "Routed",
            "n": len(records),
            "accuracy": _accuracy([r.final_correct for r in automated]),
            "coverage": len(automated) / len(records) if records else 0.0,
            "mean_latency_ms": _mean([r.routed_latency_ms for r in records]),
        },
    ]


def calibration_summary(records: list[Record], n_bins: int = 10) -> dict:
    """ECE for both of Laya's signals, next to mean confidence and accuracy."""
    correct = [r.laya_correct for r in records]
    return {
        "n": len(records),
        "accuracy": _accuracy(correct),
        "mean_confidence": _mean([r.laya_confidence for r in records]),
        "mean_top_prob": _mean([r.laya_top_prob for r in records]),
        "ece_confidence": expected_calibration_error([r.laya_confidence for r in records], correct, n_bins),
        "ece_top_prob": expected_calibration_error([r.laya_top_prob for r in records], correct, n_bins),
    }


def _fmt(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "ms":
        return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"
    return str(value)


def format_rows(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """Render rows as a fixed width text table. `columns` is (key, header, kind)."""
    cells = [[header for _, header, _ in columns]]
    cells += [[_fmt(row[key], kind) for key, _, kind in columns] for row in rows]
    widths = [max(len(line[i]) for line in cells) for i in range(len(columns))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(line, widths, strict=True)).rstrip() for line in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


TIER_COLUMNS = [
    ("tier", "tier", "str"),
    ("range", "confidence", "str"),
    ("paper", "paper says", "str"),
    ("demo", "demo does", "str"),
    ("count", "n", "str"),
    ("share", "share", "pct"),
    ("laya_acc", "Laya acc", "pct"),
    ("final_acc", "routed acc", "pct"),
    ("mean_latency_ms", "mean latency", "ms"),
]

TRADEOFF_COLUMNS = [
    ("system", "system", "str"),
    ("n", "n", "str"),
    ("accuracy", "accuracy", "pct"),
    ("coverage", "automated", "pct"),
    ("mean_latency_ms", "mean latency", "ms"),
]


def render(run: RunResult, n_bins: int = 10) -> str:
    """The full text report the CLI prints."""
    meta = run.meta
    cal = calibration_summary(run.records, n_bins)
    parts = [
        (
            f"Run {meta['started_at']}  n={len(run.records)}  Laya on {meta['device']}  "
            f"routing signal: {meta['signal']}  thresholds: {tuple(meta['thresholds'])}"
        ),
        f"System 2: {meta['gemini_status']}",
        "",
        "Per tier",
        format_rows(tier_table(run.records, run.tiers), TIER_COLUMNS),
        "",
        "Trade off",
        format_rows(tradeoff_table(run), TRADEOFF_COLUMNS),
        "",
        "Calibration of Laya",
        (
            f"  accuracy {cal['accuracy']:.1%}   mean confidence {cal['mean_confidence']:.3f}   "
            f"mean top probability {cal['mean_top_prob']:.3f}"
        ),
        (
            f"  ECE ({n_bins} bins): confidence field {cal['ece_confidence']:.3f}   "
            f"top probability {cal['ece_top_prob']:.3f}"
        ),
    ]
    return "\n".join(parts)
