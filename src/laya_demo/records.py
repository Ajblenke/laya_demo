"""The per item record and the run container that the cache saves and replays."""

from dataclasses import asdict, dataclass, field

from .routing import make_tiers


@dataclass
class Record:
    idx: int
    text: str
    gold: str
    laya_answer: str
    laya_confidence: float
    laya_top_prob: float
    laya_latency_ms: float
    tier: str = ""
    action: str = ""
    gemini_answer: str | None = None
    gemini_latency_ms: float | None = None
    gemini_error: str | None = None
    final_answer: str | None = None
    final_source: str = ""  # laya, gemini, laya_fallback, or human

    @property
    def laya_correct(self) -> bool:
        return self.laya_answer == self.gold

    @property
    def gemini_correct(self) -> bool | None:
        return None if self.gemini_answer is None else self.gemini_answer == self.gold

    @property
    def final_correct(self) -> bool | None:
        return None if self.final_answer is None else self.final_answer == self.gold

    @property
    def routed_latency_ms(self) -> float:
        """What this item cost the routed system: Laya always, plus Gemini when it was called."""
        escalated = self.action == "escalate" and self.gemini_latency_ms is not None
        return self.laya_latency_ms + (self.gemini_latency_ms if escalated else 0.0)


@dataclass
class RunResult:
    meta: dict
    labels: list[str]
    records: list[Record]
    gemini_baseline_idx: list[int] = field(default_factory=list)

    @property
    def tiers(self):
        return make_tiers(*self.meta["thresholds"])

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "labels": self.labels,
            "gemini_baseline_idx": self.gemini_baseline_idx,
            "records": [asdict(r) for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunResult:
        return cls(
            meta=data["meta"],
            labels=data["labels"],
            records=[Record(**r) for r in data["records"]],
            gemini_baseline_idx=list(data.get("gemini_baseline_idx", [])),
        )
