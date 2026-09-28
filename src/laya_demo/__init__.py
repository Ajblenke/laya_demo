"""Confidence routing demo: Laya as System 1, Gemini as System 2, humans at the bottom."""

import argparse
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    from .pipeline import SIGNALS, Config
    from .routing import PAPER_THRESHOLDS

    d = Config()
    p = argparse.ArgumentParser(prog="laya-demo", description=__doc__)
    p.add_argument("--n", type=int, default=d.n, help="items to sample from Banking77 test (default %(default)s)")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        metavar=("HIGH", "MID", "LOW"),
        default=list(PAPER_THRESHOLDS),
        help="tier cut points, highest first (default: the paper's 0.75 0.55 0.35)",
    )
    p.add_argument("--signal", choices=SIGNALS, default=d.signal, help="Laya number to route on")
    p.add_argument("--laya-model", default=d.laya_model, help="english, multilingual, or typed-decisions")
    p.add_argument("--device", default=None, help="cuda or cpu (default: whatever Laya picks)")
    p.add_argument("--no-gemini", action="store_true", help="run Laya alone and skip escalation")
    p.add_argument("--gemini-model", default=None, help="default: $GEMINI_MODEL or gemini-3.5-flash")
    p.add_argument("--max-gemini-calls", type=int, default=d.max_gemini_calls)
    p.add_argument("--gemini-baseline", type=int, default=d.gemini_baseline, help="random items for Gemini alone")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--replay", action="store_true", help="print the last cached run instead of running")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)

    from .cache import load_run, save_run
    from .report import render

    if args.replay:
        run = load_run(data_dir=args.data_dir)
    else:
        from .pipeline import Config, run_pipeline

        cfg = Config(
            n=args.n,
            seed=args.seed,
            thresholds=tuple(args.thresholds),
            signal=args.signal,
            laya_model=args.laya_model,
            device=args.device,
            use_gemini=not args.no_gemini,
            gemini_model=args.gemini_model,
            max_gemini_calls=args.max_gemini_calls,
            gemini_baseline=args.gemini_baseline,
        )
        try:
            run = run_pipeline(cfg)
        except Exception as exc:
            print(f"\nThe live run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("Replay the last good run with: uv run laya-demo --replay", file=sys.stderr)
            raise SystemExit(1) from exc
        path = save_run(run, args.data_dir)
        print(f"\nSaved run to {path} ({run.meta['duration_s']} s)")

    print()
    print(render(run))

    import matplotlib

    matplotlib.use("Agg")
    from .calibration import plot_reliability

    ax = plot_reliability([r.laya_confidence for r in run.records], [r.laya_correct for r in run.records])
    out = args.data_dir / "reliability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nReliability diagram: {out}")
