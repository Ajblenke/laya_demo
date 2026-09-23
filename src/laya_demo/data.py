"""Build the pilot, eval and train splits from the Hugging Face dataset."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from laya_demo.schema import QUEUE_MERGE

DATASET = "Tobi-Bueck/customer-support-tickets"
DATA_DIR = Path("data")
SEED = 20260923
EVAL_SIZE = 1000
PILOT_SIZE = 100


def split_path(split: str) -> Path:
    return DATA_DIR / f"{split}.jsonl"


def load_split(split: str) -> list[dict]:
    with split_path(split).open() as f:
        return [json.loads(line) for line in f]


def normalize(rows) -> list[dict]:
    """English only, one row per distinct body, labels mapped to the demo schema."""
    seen: set[str] = set()
    out = []
    for idx, r in enumerate(rows):
        body = (r["body"] or "").strip()
        if r["language"] != "en" or not body or body in seen:
            continue
        seen.add(body)
        out.append(
            {
                "id": f"t{idx}",
                "subject": r["subject"] or "",
                "body": body,
                "queue": QUEUE_MERGE[r["queue"]],
                "queue_raw": r["queue"],
                "type": r["type"].lower(),
                "priority": r["priority"],
            }
        )
    return out


def balanced_take(pool: dict[str, list[dict]], n: int) -> list[dict]:
    """Take n rows spread as evenly as possible across classes, popping them from the pool."""
    classes = sorted(pool)
    base, extra = divmod(n, len(classes))
    taken = []
    for i, c in enumerate(classes):
        k = base + (1 if i < extra else 0)
        if len(pool[c]) < k:
            raise ValueError(f"class {c!r} has {len(pool[c])} rows, need {k}")
        taken.extend(pool[c][:k])
        del pool[c][:k]
    return taken


def make_splits(rows: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    pool: dict[str, list[dict]] = defaultdict(list)
    for r in shuffled:
        pool[r["queue"]].append(r)
    pilot = balanced_take(pool, PILOT_SIZE)
    evaluation = balanced_take(pool, EVAL_SIZE)
    train = [r for c in sorted(pool) for r in pool[c]]
    for split in (pilot, evaluation, train):
        rng.shuffle(split)
    return {"pilot": pilot, "eval": evaluation, "train": train}


def prepare() -> dict[str, int]:
    from datasets import load_dataset

    rows = normalize(load_dataset(DATASET, split="train"))
    DATA_DIR.mkdir(exist_ok=True)
    sizes = {}
    for name, split in make_splits(rows).items():
        with split_path(name).open("w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        sizes[name] = len(split)
    return sizes
