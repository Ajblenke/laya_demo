"""Confidence tiers, adapted from Nandakishor M, arXiv 2510.01237.

The paper assesses confidence before generation and routes on four tiers:

    >= 0.75        answer locally
    0.55 to 0.75   retrieval augmented generation
    0.35 to 0.55   a larger model
    < 0.35         human review

This demo has no retrieval step, so both middle tiers escalate to Gemini as System 2.
They stay separate tiers so the table still shows how the paper would split them.
"""

import math
from dataclasses import dataclass

PAPER_THRESHOLDS = (0.75, 0.55, 0.35)

ACCEPT = "accept"
ESCALATE = "escalate"
HUMAN = "human"


@dataclass(frozen=True)
class Tier:
    """A confidence band. An item belongs to the first tier whose `lower` bound it meets."""

    name: str
    lower: float
    action: str
    paper_action: str


def make_tiers(
    high: float = PAPER_THRESHOLDS[0],
    mid: float = PAPER_THRESHOLDS[1],
    low: float = PAPER_THRESHOLDS[2],
) -> tuple[Tier, ...]:
    """Build the four tiers from three thresholds, highest first."""
    if not 0.0 <= low <= mid <= high <= 1.0:
        raise ValueError(f"thresholds must satisfy 0 <= low <= mid <= high <= 1, got {high}, {mid}, {low}")
    return (
        Tier("local", high, ACCEPT, "answer locally"),
        Tier("retrieval", mid, ESCALATE, "retrieval"),
        Tier("larger_model", low, ESCALATE, "larger model"),
        Tier("human", 0.0, HUMAN, "human review"),
    )


def assign_tier(confidence: float, tiers: tuple[Tier, ...]) -> Tier:
    """Return the tier for one confidence. A missing or NaN confidence goes to the last tier."""
    if confidence is None or math.isnan(confidence):
        return tiers[-1]
    for tier in tiers:
        if confidence >= tier.lower:
            return tier
    return tiers[-1]


def tier_bounds(tiers: tuple[Tier, ...]) -> dict[str, str]:
    """Human readable range for each tier, for example `0.55 to 0.75`."""
    out = {}
    upper = None
    for tier in tiers:
        if upper is None:
            out[tier.name] = f">= {tier.lower:.2f}"
        elif tier.lower <= 0.0:
            out[tier.name] = f"< {upper:.2f}"
        else:
            out[tier.name] = f"{tier.lower:.2f} to {upper:.2f}"
        upper = tier.lower
    return out
