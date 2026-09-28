"""Expected calibration error and a reliability diagram.

A confidence is calibrated when, among the answers given with confidence near c, a fraction
near c are correct. Bins are equal width on [0, 1]; the first bin is closed on both ends and
the rest are (lo, hi], which matches `laya.common.ece_score`.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_bins(confidence, correct, n_bins: int = 10) -> list[Bin]:
    """Group answers into equal width confidence bins. Empty bins are left out."""
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.shape != hit.shape:
        raise ValueError("confidence and correct must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i, (lo, hi) in enumerate(pairwise(edges)):
        sel = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if sel.any():
            bins.append(Bin(float(lo), float(hi), int(sel.sum()), float(conf[sel].mean()), float(hit[sel].mean())))
    return bins


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """Weighted mean gap between confidence and accuracy across bins. NaN for no data."""
    bins = reliability_bins(confidence, correct, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return float("nan")
    return float(sum(b.count / total * abs(b.accuracy - b.mean_confidence) for b in bins))


def plot_reliability(confidence, correct, n_bins: int = 10, ax=None, title: str | None = None, label: str = "Laya"):
    """Draw a reliability diagram: accuracy per confidence bin against the diagonal."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    bins = reliability_bins(confidence, correct, n_bins)
    width = 1.0 / n_bins
    ece = expected_calibration_error(confidence, correct, n_bins)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfectly calibrated")
    ax.bar(
        [b.lo for b in bins],
        [b.accuracy for b in bins],
        width=width,
        align="edge",
        edgecolor="black",
        color="#4C78A8",
        alpha=0.85,
        label=f"{label} accuracy per bin",
    )
    ax.bar(
        [b.lo for b in bins],
        [b.mean_confidence - b.accuracy for b in bins],
        bottom=[b.accuracy for b in bins],
        width=width,
        align="edge",
        color="#E45756",
        alpha=0.35,
        hatch="//",
        edgecolor="#E45756",
        label="gap to stated confidence",
    )
    for b in bins:
        ax.text(b.lo + width / 2, 0.02, str(b.count), ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(title or f"{label} reliability, ECE = {ece:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    return ax
