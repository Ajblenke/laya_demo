"""Load a small labeled sample of Banking77 for the demo."""

from dataclasses import dataclass

# The canonical PolyAI/banking77 repo ships a loading script, which `datasets` 5 refuses to run.
# legacy-datasets/banking77 is the same data (10,003 train, 3,080 test, 77 labels) as parquet.
DATASET = "legacy-datasets/banking77"


@dataclass(frozen=True)
class Item:
    """One labeled question: its position in the sample, the customer text, and the gold label."""

    idx: int
    text: str
    label: str


def readable(label: str) -> str:
    """Turn a Banking77 label such as `card_arrival` into the words `card arrival`.

    Laya squeezes 77 options into a head budget of about 3 tokens each, so words tokenize
    into a more useful prefix than snake case does.
    """
    return label.replace("_", " ").replace("?", "").strip()


def load_banking77(n: int = 150, seed: int = 0, split: str = "test") -> tuple[list[Item], list[str]]:
    """Return a seeded random sample of `n` items and the full ordered label list."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    names = list(ds.features["label"].names)
    n = min(n, len(ds))
    sample = ds.shuffle(seed=seed).select(range(n))
    items = [Item(i, row["text"], names[row["label"]]) for i, row in enumerate(sample)]
    return items, names
