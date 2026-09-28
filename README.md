# Laya confidence routing demo

A classroom demo for a Trustworthy AI talk.
It turns the routing ladder from Nandakishor M, *Confidence-Aware Routing for LLM Reliability Enhancement* (arXiv 2510.01237), into running code, with Laya as System 1 and Gemini as System 2.

The point: routing by confidence only works if the confidence is honest.

## What it does

1. Laya (`laya` 0.3.10, the `english` checkpoint) answers a seeded sample of Banking77 test messages as one 77 option `choice` question, recording its answer, its `confidence` field, its top probability, and latency.
2. Each item is routed on the paper's tiers:

   | tier | confidence | paper | demo |
   |---|---|---|---|
   | local | >= 0.75 | answer locally | accept Laya's answer |
   | retrieval | 0.55 to 0.75 | retrieval | escalate to Gemini |
   | larger_model | 0.35 to 0.55 | larger model | escalate to Gemini |
   | human | < 0.35 | human review | flag for a person |

   The demo has no retrieval step, so both middle tiers go to Gemini.
   The thresholds are parameters.
3. It reports each tier's count, Laya accuracy, routed accuracy, and mean latency, then compares Laya alone, Gemini alone, and the routed system.
4. It checks calibration: expected calibration error (ECE) and a reliability diagram for Laya's confidences.

Two details from the installed Laya source that matter for the talk:

- Laya's `confidence` is not the probability of its answer.
  It is one minus normalized entropy, `1 - H(p) / log(k)`.
  The demo records both and reports ECE for each.
- The `english` checkpoint ships a temperature of 0.1006 for choice questions with more than 10 options, which would sharpen the probabilities tenfold.
  Laya clamps it to 0.5 at load and warns that the affected confidences are uncalibrated.
  Banking77 is in exactly that bucket.

## Setup

```bash
uv sync
cp .env.example .env
```

Put a Gemini API key in `.env` as `gemini_api_key=...` (get one at https://aistudio.google.com/apikey).
Without a key the demo still runs: Laya answers, items are routed, escalated items keep Laya's answer, and the report says escalation was skipped.

The first run downloads the Laya checkpoint (about 800 MB) and Banking77 into the Hugging Face cache.
Run it once while online before class.

## Run it

```bash
uv run laya-demo                        # 150 items, paper thresholds, up to 60 Gemini calls
uv run laya-demo --n 80 --device cpu    # a laptop with no GPU
uv run laya-demo --thresholds 0.9 0.6 0.3 --signal top_prob
uv run laya-demo --no-gemini
uv run laya-demo --replay               # print the last saved run, no network
```

Each run is saved to `data/runs/` and copied to `data/last_run.json`.
The reliability diagram is written to `data/reliability.png`.

Gemini calls are capped by `--max-gemini-calls`.
Escalated items get the budget first, then a seeded random sample (`--gemini-baseline`, default 30) so the Gemini alone number is not skewed toward hard items.
The model defaults to `gemini-3.5-flash`; change it with `--gemini-model` or `GEMINI_MODEL`.

"Routed accuracy" counts only the items a model answered.
Items flagged for human review are reported as coverage, not scored.

## The notebook

`intro_to_laya.ipynb` is the presenter walkthrough.
Select the project's `.venv` as its kernel.
It ends on the tier table and the reliability diagram.
If the dataset, the model, or the run fails, it switches to the last saved run on its own; set `LIVE = False` in the first cell to replay on purpose.

## Timing

Measured on 2026-09-28 with 150 items:

- RTX 5060: about 20 ms per item, under 10 seconds for the Laya pass.
- CPU only: about 680 ms per item, 107 seconds for the Laya pass.

## Tests

```bash
uv run pytest
uv run ruff check
uv run ruff format --check
```

The tests need no network and no model weights.

## Sources

- Laya: https://github.com/NandhaKishorM/laya (Apache 2.0), PyPI `laya`
- Paper: https://arxiv.org/abs/2510.01237
- Banking77: `legacy-datasets/banking77` on the Hugging Face Hub, a parquet copy of `PolyAI/banking77`, whose loading script `datasets` 5 no longer runs
