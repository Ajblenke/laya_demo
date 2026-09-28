"""Confidence routing demo: Laya as System 1, Gemini as System 2, humans at the bottom.

Commands:
  laya-demo    the whole run in one pass, printing the report at the end
  laya-run     Laya alone, streaming each answer; saves data/laya_results.json
  gemini-run   Gemini on Laya's escalated items, streaming each batch; saves a full run
  laya-race    Laya and Gemini on the same messages at once, side by side, live
  laya-report  the accuracy and calibration report for the last saved run, no network
"""

import argparse
import sys
import time
from pathlib import Path


def _add_sample_args(p: argparse.ArgumentParser, n: int) -> None:
    from .pipeline import Config

    d = Config()
    p.add_argument("--n", type=int, default=n, help="items to sample from Banking77 test (default %(default)s)")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--laya-model", default=d.laya_model, help="english, multilingual, or typed-decisions")
    p.add_argument("--device", default=None, help="cuda or cpu (default: whatever Laya picks)")


def _add_routing_args(p: argparse.ArgumentParser) -> None:
    from .pipeline import SIGNALS, Config
    from .routing import PAPER_THRESHOLDS

    p.add_argument(
        "--thresholds",
        type=float,
        nargs=3,
        metavar=("HIGH", "MID", "LOW"),
        default=list(PAPER_THRESHOLDS),
        help="tier cut points, highest first (default: the paper's 0.75 0.55 0.35)",
    )
    p.add_argument("--signal", choices=SIGNALS, default=Config().signal, help="Laya number to route on")


def _add_gemini_args(p: argparse.ArgumentParser, budget: bool = True) -> None:
    from .pipeline import Config

    d = Config()
    p.add_argument("--gemini-model", default=None, help="default: $GEMINI_MODEL or gemini-3.5-flash")
    if budget:
        p.add_argument("--max-gemini-calls", type=int, default=d.max_gemini_calls)
        p.add_argument("--gemini-baseline", type=int, default=d.gemini_baseline, help="random items for Gemini alone")


def _add_data_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=Path("data"))


def _parser() -> argparse.ArgumentParser:
    from .pipeline import Config

    p = argparse.ArgumentParser(prog="laya-demo", description=__doc__.splitlines()[0])
    _add_sample_args(p, Config().n)
    _add_routing_args(p)
    p.add_argument("--no-gemini", action="store_true", help="run Laya alone and skip escalation")
    _add_gemini_args(p)
    _add_data_dir(p)
    p.add_argument("--replay", action="store_true", help="print the last cached run instead of running")
    return p


def _write_reliability(run, data_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from .calibration import plot_reliability

    ax = plot_reliability([r.laya_confidence for r in run.records], [r.laya_correct for r in run.records])
    out = data_dir / "reliability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    return out


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
    print(f"\nReliability diagram: {_write_reliability(run, args.data_dir)}")


def _ms(value: float) -> str:
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def laya_main(argv: list[str] | None = None) -> None:
    """Run Laya alone, printing each answer as it lands, and save the routed results for `gemini-run`."""
    from .pipeline import Config

    p = argparse.ArgumentParser(prog="laya-run", description=laya_main.__doc__)
    _add_sample_args(p, Config().n)
    _add_routing_args(p)
    _add_data_dir(p)
    args = p.parse_args(argv)

    from collections import Counter

    from rich.console import Console

    from .cache import save_laya_results
    from .data import load_banking77
    from .pipeline import GeminiOutcome, build_meta, check_signal, load_system1, now_iso, route_records, run_laya
    from .records import RunResult
    from .routing import assign_tier, make_tiers

    cfg = Config(
        n=args.n,
        seed=args.seed,
        thresholds=tuple(args.thresholds),
        signal=args.signal,
        laya_model=args.laya_model,
        device=args.device,
    )
    check_signal(cfg.signal)
    tiers = make_tiers(*cfg.thresholds)
    console = Console(highlight=False)
    started = now_iso()

    console.print(f"Loading Banking77 sample (n={cfg.n}, seed={cfg.seed})")
    items, labels = load_banking77(cfg.n, cfg.seed)
    laya = load_system1(cfg, labels, log=console.print)
    console.rule(f"Laya ({cfg.laya_model}, {laya.device}) answering {len(items)} messages")

    def show(r) -> None:
        value = r.laya_confidence if cfg.signal == "confidence" else r.laya_top_prob
        tier = assign_tier(value, tiers).name
        console.print(
            f"#{r.idx:<4}[dim]{_ms(r.laya_latency_ms):>9}[/]  conf [cyan]{r.laya_confidence:.2f}[/]  "
            f"top_p {r.laya_top_prob:.2f}  {tier:<12}  [bold]{r.laya_answer}[/]",
            soft_wrap=True,
        )

    t0 = time.perf_counter()
    records = run_laya(items, laya, on_item=show)
    elapsed = time.perf_counter() - t0
    route_records(records, tiers, cfg.signal)

    waiting = GeminiOutcome(cfg.gemini_model or "-", "not run yet: run `uv run gemini-run`")
    meta = build_meta(cfg, started, elapsed, laya.device, waiting, len(records))
    meta["laya_s"] = round(elapsed, 2)
    path = save_laya_results(RunResult(meta, labels, records), args.data_dir)

    counts = Counter(r.tier for r in records)
    console.rule()
    console.print(
        f"[bold green]Laya answered {len(records)} messages in {elapsed:.2f} s "
        f"({_ms(elapsed * 1000 / max(len(records), 1))} each).[/]"
    )
    console.print("Tiers: " + ", ".join(f"{t.name} {counts[t.name]}" for t in tiers))
    console.print(f"Saved to {path}. Next: [bold]uv run gemini-run[/]")


def gemini_main(argv: list[str] | None = None) -> None:
    """Send Laya's escalated items to Gemini, showing each batch request live, and save the full run."""
    p = argparse.ArgumentParser(prog="gemini-run", description=gemini_main.__doc__)
    _add_gemini_args(p)
    _add_data_dir(p)
    args = p.parse_args(argv)

    from rich.console import Console
    from rich.progress import Progress, ProgressColumn, SpinnerColumn, TextColumn
    from rich.text import Text

    from .cache import load_laya_results, save_run
    from .pipeline import Config, finalize, now_iso, plan_gemini_calls, run_gemini
    from .records import RunResult
    from .system2 import GeminiSystem2, default_model, load_api_key

    console = Console(highlight=False)
    laya_run = load_laya_results(args.data_dir)
    meta = laya_run.meta
    cfg = Config(
        n=meta["n"],
        seed=meta["seed"],
        thresholds=tuple(meta["thresholds"]),
        signal=meta["signal"],
        laya_model=meta["laya_model"],
        gemini_model=args.gemini_model,
        max_gemini_calls=args.max_gemini_calls,
        gemini_baseline=args.gemini_baseline,
    )
    key = load_api_key()
    if key is None:
        console.print("[red]No gemini_api_key in .env or the environment.[/] See the README's Setup section.")
        raise SystemExit(1)
    system2 = GeminiSystem2(laya_run.labels, key, model=args.gemini_model or default_model())
    records = laya_run.records
    for r in records:  # a rerun starts from Laya's answers, not the previous Gemini pass
        r.gemini_answer = r.gemini_latency_ms = r.gemini_error = None

    class Seconds(ProgressColumn):
        def render(self, task) -> Text:
            return Text(f"{task.elapsed or 0.0:6.1f} s", style="yellow")

    # The same plan run_gemini makes, so each batch's answers can be printed against their items.
    planned, _ = plan_gemini_calls(records, cfg.gemini_baseline, cfg.max_gemini_calls, cfg.seed)
    by_idx = {r.idx: r for r in records}
    answered = 0
    started = now_iso()
    t0 = time.perf_counter()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), Seconds(), console=console) as progress:
        tasks: dict[int, int] = {}

        def on_batch_start(batch_no: int, n_batches: int, size: int) -> None:
            tasks[batch_no] = progress.add_task(f"batch {batch_no}/{n_batches}: {size} messages in flight", total=1)

        def on_batch_done(batch_no: int, n_batches: int, answers, model: str | None) -> None:
            nonlocal answered
            task = next(t for t in progress.tasks if t.id == tasks[batch_no])
            seconds = task.elapsed or 0.0
            progress.update(tasks[batch_no], completed=1, visible=False)
            per_item = seconds * 1000 / max(len(answers), 1)
            fallback = " (fallback)" if model and model != system2.model else ""
            progress.console.print(
                f"[bold]batch {batch_no}/{n_batches}[/] answered by {model or 'no model'}{fallback} "
                f"in [yellow]{seconds:.1f} s[/] ({_ms(per_item)} per message)"
            )
            for i, a in zip(planned[answered : answered + len(answers)], answers, strict=True):
                r = by_idx[i]
                shown = f"[bold]{a.answer}[/]" if a.answer else f"[red]{a.error}[/]"
                progress.console.print(
                    f"  #{i:<4}{r.tier:<12}  Laya {r.laya_answer:<34}  Gemini {shown}", soft_wrap=True
                )
            answered += len(answers)

        console.rule(f"Gemini ({system2.model})")
        outcome = run_gemini(
            records,
            laya_run.labels,
            cfg,
            log=console.print,
            system2=system2,
            on_batch_start=on_batch_start,
            on_batch_done=on_batch_done,
        )
    elapsed = time.perf_counter() - t0
    finalize(records)

    meta = {
        **meta,
        "started_at": started,
        "laya_started_at": meta["started_at"],
        "duration_s": round(meta["duration_s"] + elapsed, 1),
        "gemini_s": round(elapsed, 2),
        "gemini_model": outcome.model,
        "gemini_status": outcome.status,
        "max_gemini_calls": cfg.max_gemini_calls,
    }
    run = RunResult(meta, laya_run.labels, records, outcome.baseline_idx)
    path = save_run(run, args.data_dir)

    console.rule()
    laya_ms = meta.get("laya_s", 0) * 1000 / max(len(records), 1)
    if answered:
        console.print(
            f"[bold green]Gemini answered {answered} messages in {elapsed:.1f} s "
            f"({_ms(elapsed * 1000 / answered)} each); "
            f"Laya took {_ms(laya_ms)} each.[/]"
        )
    console.print(f"System 2: {outcome.status}")
    console.print(f"Saved run to {path}. Next: [bold]uv run laya-report[/]")


def race_main(argv: list[str] | None = None) -> None:
    """Race Laya and Gemini on the same messages, side by side and live, then save the run."""
    p = argparse.ArgumentParser(prog="laya-race", description=race_main.__doc__)
    _add_sample_args(p, 60)
    _add_routing_args(p)
    _add_gemini_args(p, budget=False)
    p.add_argument("--no-gemini", action="store_true", help="race Laya alone")
    p.add_argument("--no-wait", action="store_true", help="start without waiting for Enter")
    _add_data_dir(p)
    args = p.parse_args(argv)

    from rich.console import Console

    from .cache import save_run
    from .data import load_banking77
    from .pipeline import Config, check_signal, load_system1, n_requests
    from .race import run_race
    from .system2 import GeminiSystem2, default_model, load_api_key

    cfg = Config(
        n=args.n,
        seed=args.seed,
        thresholds=tuple(args.thresholds),
        signal=args.signal,
        laya_model=args.laya_model,
        device=args.device,
        use_gemini=not args.no_gemini,
        gemini_model=args.gemini_model,
        max_gemini_calls=args.n,
        gemini_baseline=args.n,
    )
    check_signal(cfg.signal)
    console = Console(highlight=False)
    console.print(f"Loading Banking77 sample (n={cfg.n}, seed={cfg.seed})")
    items, labels = load_banking77(cfg.n, cfg.seed)
    laya = load_system1(cfg, labels, log=console.print)

    system2 = None
    key = load_api_key() if cfg.use_gemini else None
    if key is not None:
        system2 = GeminiSystem2(labels, key, model=cfg.gemini_model or default_model())
    elif cfg.use_gemini:
        console.print("[yellow]No gemini_api_key in .env; racing Laya alone.[/]")

    gemini_plan = (
        f"Gemini sends them in {n_requests(len(items), cfg.gemini_batch_size)} requests of up to "
        f"{cfg.gemini_batch_size}"
        if system2
        else "Gemini is skipped"
    )
    console.print(f"Ready: {len(items)} messages. Laya answers one at a time; {gemini_plan}.")
    if not args.no_wait and sys.stdin.isatty():
        input("Press Enter to start the race ")

    run = run_race(cfg, items, labels, laya, system2, console=console)
    path = save_run(run, args.data_dir)
    console.print(f"\nSaved run to {path}. Next: [bold]uv run laya-report[/]")


def report_main(argv: list[str] | None = None) -> None:
    """Print the accuracy and calibration report for the last saved run, with no network."""
    p = argparse.ArgumentParser(prog="laya-report", description=report_main.__doc__)
    _add_data_dir(p)
    p.add_argument("--run", type=Path, default=None, help="a saved run file (default: data/last_run.json)")
    p.add_argument("--no-items", action="store_true", help="leave out the per item table")
    args = p.parse_args(argv)

    from .cache import load_run
    from .report import render, render_items

    run = load_run(args.run, data_dir=args.data_dir)
    print(render(run))
    if not args.no_items:
        print("\nPer item, most confident first")
        print(render_items(run))
    print(f"\nReliability diagram: {_write_reliability(run, args.data_dir)}")
