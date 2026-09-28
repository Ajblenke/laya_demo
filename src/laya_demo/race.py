"""A live race: Laya and Gemini classify the same messages at the same time, side by side.

Laya runs item by item in one thread; Gemini runs batch by batch in another, one request per
batch, as the free tier requires. The display shows each side's progress and answers as they
land, and holds back correctness so accuracy can be discussed afterward with `laya-report`.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .pipeline import (
    Config,
    GeminiOutcome,
    apply_gemini,
    build_meta,
    finalize,
    gemini_status,
    gemini_unavailable,
    now_iso,
    route_records,
    run_laya,
)
from .records import Record, RunResult
from .routing import make_tiers

RECENT = 12  # answers shown in each lane


@dataclass
class Lane:
    """One side of the race. Worker threads write to it; the display reads it under the lock."""

    title: str
    total: int
    done: int = 0
    started: float | None = None
    finished: float | None = None
    waiting: str | None = None  # what the lane is waiting on, for example "request 1/2 in flight"
    waiting_since: float | None = None
    note: str | None = None  # a remark under the progress line, for example which fallback model answered
    skipped: bool = False  # the lane never runs, for example Gemini with no key
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT))
    spinner: Spinner = field(default_factory=lambda: Spinner("dots", style="yellow"))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, answer: str | None, ms: float, conf: float | None = None) -> None:
        with self.lock:
            self.done += 1
            self.recent.append((answer, ms, conf))

    def elapsed(self, now: float) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or now) - self.started


def _ms(value: float) -> str:
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def _line(text: str, style: str = "") -> Text:
    """One display line that never wraps, so a narrow terminal cuts it with an ellipsis instead."""
    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


def render_lane(lane: Lane, now: float) -> Panel:
    with lane.lock:
        elapsed = lane.elapsed(now)
        rate = lane.done / elapsed if elapsed > 0 else 0.0
        rows: list = [
            # A bare ProgressBar ends without a newline inside a Group; Padding renders it as a full line.
            Padding(ProgressBar(total=max(lane.total, 1), completed=lane.done), 0),
            _line(f"{lane.done}/{lane.total} done   {elapsed:5.1f} s   {rate:5.1f} per s", style="bold"),
        ]
        if lane.waiting and lane.finished is None:
            wait = now - (lane.waiting_since or now)
            line = _line("")
            line.append_text(lane.spinner.render(now))
            line.append(f" {lane.waiting}  {wait:5.1f} s", style="yellow")
            rows.append(line)
        else:
            rows.append(_line(lane.note or "", style="yellow"))
        rows.append(Text(""))
        for answer, ms, conf in lane.recent:
            line = _line(f"{_ms(ms):>9}  ")
            if conf is not None:
                line.append(f"{conf:.2f}  ", style="cyan")
            line.append(answer or "error", style="bold" if answer else "red")
            rows.append(line)
        rows += [Text("")] * (RECENT - len(lane.recent))  # a fixed height, so the panels do not jump
        done = lane.finished is not None
        subtitle = f"done in {elapsed:.1f} s" if done else None
    return Panel(Group(*rows), title=lane.title, subtitle=subtitle, border_style="green" if done else "blue")


def verdict(laya: Lane, gemini: Lane, now: float) -> Text:
    """The footer: who is done, and once both are, how much faster Laya was per item."""
    if laya.finished is None or (gemini.finished is None and not gemini.skipped):
        return Text("")
    lt = laya.elapsed(now)
    if gemini.done == 0:
        return Text(f"Laya answered {laya.done} messages in {lt:.1f} s.", style="bold")
    gt = gemini.elapsed(now)
    per_laya, per_gemini = lt / max(laya.done, 1), gt / max(gemini.done, 1)
    speedup = per_gemini / per_laya if per_laya > 0 else float("inf")
    return Text(
        f"Laya: {laya.done} in {lt:.1f} s ({_ms(per_laya * 1000)} each).   "
        f"Gemini: {gemini.done} in {gt:.1f} s ({_ms(per_gemini * 1000)} each).   "
        f"Laya was {speedup:,.0f}x faster per message.",
        style="bold green",
    )


class RaceView:
    """A renderable that `rich.live.Live` redraws on every refresh."""

    def __init__(self, laya: Lane, gemini: Lane):
        self.laya, self.gemini = laya, gemini

    def __rich__(self):
        now = time.perf_counter()
        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(render_lane(self.laya, now), render_lane(self.gemini, now))
        return Group(grid, verdict(self.laya, self.gemini, now))


def run_race(
    cfg: Config, items, labels: list[str], laya, system2=None, display: bool = True, console: Console | None = None
) -> RunResult:
    """Race Laya against Gemini on `items`, then route and finalize as a normal run.

    Gemini answers every item, not only the escalated ones, so the race is head to head and
    "Gemini alone" is scored on the whole sample. With `system2=None` only Laya runs.
    """
    started = now_iso()
    laya_lane = Lane(f"Laya ({cfg.laya_model}, {laya.device})", len(items))
    gemini_lane = Lane(f"Gemini ({system2.model})" if system2 else "Gemini", len(items) if system2 else 0)
    records: list[Record] = []
    gemini_answers: list = []
    failures: list[Exception] = []

    def laya_work():
        try:
            records.extend(
                run_laya(
                    items, laya, on_item=lambda r: laya_lane.add(r.laya_answer, r.laya_latency_ms, r.laya_confidence)
                )
            )
        # Broad on purpose: a worker must hand any failure to the main thread, which re-raises it.
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)
        finally:
            laya_lane.finished = time.perf_counter()

    def on_batch_start(batch_no: int, n_batches: int, size: int) -> None:
        with gemini_lane.lock:
            gemini_lane.waiting = f"request {batch_no}/{n_batches} in flight"
            gemini_lane.waiting_since = time.perf_counter()

    def on_batch_done(batch_no: int, n_batches: int, answers, model: str | None) -> None:
        for a in answers:
            gemini_lane.add(a.answer, a.latency_ms)
        if model and model != system2.model:
            with gemini_lane.lock:
                gemini_lane.note = f"answered by {model} (fallback)"

    def gemini_work():
        try:
            gemini_answers.extend(
                system2.answer_many(
                    [item.text for item in items],
                    batch_size=cfg.gemini_batch_size,
                    on_batch_start=on_batch_start,
                    on_batch_done=on_batch_done,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)
        finally:
            gemini_lane.finished = time.perf_counter()

    workers = [threading.Thread(target=laya_work, name="laya")]
    if system2 is not None:
        workers.append(threading.Thread(target=gemini_work, name="gemini"))
    else:
        gemini_lane.note, gemini_lane.skipped = "no gemini_api_key in .env, skipped", True

    t0 = time.perf_counter()
    laya_lane.started = gemini_lane.started = t0
    for w in workers:
        w.start()
    if display:
        from rich.live import Live

        view = RaceView(laya_lane, gemini_lane)
        with Live(view, console=console, refresh_per_second=12):
            while any(w.is_alive() for w in workers):
                time.sleep(0.05)
    for w in workers:
        w.join()
    if failures:
        raise failures[0]

    route_records(records, make_tiers(*cfg.thresholds), cfg.signal)
    if system2 is not None:
        apply_gemini(records, {item.idx: a for item, a in zip(items, gemini_answers, strict=True)})
        status = gemini_status(system2.model, system2, gemini_answers, cfg.gemini_batch_size)
        gemini = GeminiOutcome(system2.model, status + "; race: Gemini answered every item", [i.idx for i in items])
    else:
        gemini = GeminiOutcome(cfg.gemini_model or "-", gemini_unavailable(True, None))
    finalize(records)

    end = time.perf_counter()
    meta = build_meta(cfg, started, end - t0, laya.device, gemini, len(records))
    meta["race"] = {
        "laya_s": round(laya_lane.elapsed(end), 2),
        "gemini_s": round(gemini_lane.elapsed(end), 2) if system2 is not None else None,
    }
    return RunResult(meta, labels, records, gemini.baseline_idx)
