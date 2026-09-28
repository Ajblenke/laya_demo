"""The split scripts (laya-run, gemini-run, laya-report) and the race, with fakes for both models."""

import io
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import laya_demo
from laya_demo import pipeline, system2
from laya_demo.cache import load_laya_results, load_run, save_laya_results
from laya_demo.data import Item
from laya_demo.pipeline import Config, route_records, run_laya
from laya_demo.race import run_race
from laya_demo.records import RunResult
from laya_demo.routing import make_tiers
from laya_demo.system1 import System1Answer
from laya_demo.system2 import GeminiSystem2

LABELS = ["a", "b", "c"]
# text: (gold, Laya's answer, Laya's confidence). Gemini always answers the gold label.
SAMPLE = {
    "sure and right": ("a", "a", 0.95),
    "sure and wrong": ("b", "a", 0.90),
    "unsure": ("c", "b", 0.60),
    "very unsure": ("c", "a", 0.40),
    "lost": ("a", "c", 0.10),
}
ITEMS = [Item(i, text, gold) for i, (text, (gold, _, _)) in enumerate(SAMPLE.items())]


class FakeLaya:
    labels, model, device = LABELS, "english", "cpu"

    def warm_up(self):
        pass

    def answer(self, text):
        _, answer, conf = SAMPLE[text]
        return System1Answer(answer, conf, conf, 1.0)


def fake_gemini(batch_size_seen: list | None = None) -> GeminiSystem2:
    """A GeminiSystem2 whose client reads the numbered prompt and answers each message's gold label."""

    def generate_content(contents, model, **_):
        texts = re.findall(r"^\[(\d+)\] (.*)$", contents, flags=re.MULTILINE)
        if batch_size_seen is not None:
            batch_size_seen.append(len(texts))
        return SimpleNamespace(text=json.dumps([{"id": int(i), "label": SAMPLE[t][0]} for i, t in texts]))

    g = GeminiSystem2.__new__(GeminiSystem2)
    g.labels, g.model, g._config = LABELS, "fake", None
    g.fallbacks, g.answered_by, g._exhausted, g.last_model = [], Counter(), set(), None
    g._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    g._sleep = lambda _: None
    return g


def laya_results() -> RunResult:
    records = run_laya(ITEMS, FakeLaya())
    route_records(records, make_tiers(), "confidence")
    meta = pipeline.build_meta(
        Config(n=len(ITEMS)),
        "2026-09-28T10:00:00",
        0.01,
        "cpu",
        pipeline.GeminiOutcome("-", "not run yet"),
        len(records),
    )
    meta["laya_s"] = 0.01
    return RunResult(meta, LABELS, records)


def test_run_laya_reports_each_item_in_order():
    seen = []
    records = run_laya(ITEMS, FakeLaya(), on_item=seen.append)
    assert [r.idx for r in seen] == [0, 1, 2, 3, 4]
    assert seen == records


def test_answer_many_reports_each_batch_start_and_finish():
    events = []
    g = fake_gemini()
    g.answer_many(
        list(SAMPLE),
        batch_size=2,
        on_batch_start=lambda b, n, size: events.append(("start", b, n, size)),
        on_batch_done=lambda b, n, answers, model: events.append(("done", b, n, len(answers), model)),
    )
    assert events == [
        ("start", 1, 3, 2),
        ("done", 1, 3, 2, "fake"),
        ("start", 2, 3, 2),
        ("done", 2, 3, 2, "fake"),
        ("start", 3, 3, 1),
        ("done", 3, 3, 1, "fake"),
    ]


def test_laya_results_round_trip(tmp_path):
    run = laya_results()
    save_laya_results(run, tmp_path)
    assert load_laya_results(tmp_path).to_dict() == run.to_dict()


def test_gemini_run_answers_escalated_items_and_saves_a_full_run(tmp_path, monkeypatch, capsys):
    save_laya_results(laya_results(), tmp_path)
    monkeypatch.setattr(system2, "load_api_key", lambda: "placeholder-for-test")
    monkeypatch.setattr(system2, "GeminiSystem2", lambda labels, key, model=None: fake_gemini())
    laya_demo.gemini_main(["--data-dir", str(tmp_path), "--gemini-baseline", "0"])

    run = load_run(data_dir=tmp_path)
    by_text = {r.text: r for r in run.records}
    assert by_text["sure and wrong"].final_source == "laya"  # confidently wrong stays wrong
    assert by_text["unsure"].final_source == "gemini" and by_text["unsure"].final_correct
    assert by_text["very unsure"].final_source == "gemini" and by_text["very unsure"].final_correct
    assert by_text["lost"].final_source == "human"
    assert "laya_started_at" in run.meta and run.meta["gemini_s"] >= 0
    assert "batch 1/1" in capsys.readouterr().out


def test_gemini_run_without_a_key_stops(tmp_path, monkeypatch):
    save_laya_results(laya_results(), tmp_path)
    monkeypatch.setattr(system2, "load_api_key", lambda: None)
    with pytest.raises(SystemExit):
        laya_demo.gemini_main(["--data-dir", str(tmp_path)])


def test_laya_run_streams_and_saves_the_handoff(tmp_path, monkeypatch, capsys):
    from laya_demo import data

    monkeypatch.setattr(data, "load_banking77", lambda n, seed: (ITEMS, LABELS))
    monkeypatch.setattr(pipeline, "load_system1", lambda cfg, labels, system1=None, log=print: FakeLaya())
    laya_demo.laya_main(["--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count("conf ") == len(ITEMS)
    assert [r.tier for r in load_laya_results(tmp_path).records] == [
        "local",
        "local",
        "retrieval",
        "larger_model",
        "human",
    ]


@pytest.mark.parametrize("display", [False, True])
def test_race_has_gemini_answer_every_item(display):
    sizes: list[int] = []
    console = Console(file=io.StringIO(), width=120, force_terminal=True)
    cfg = Config(n=len(ITEMS), gemini_batch_size=2)
    run = run_race(cfg, ITEMS, LABELS, FakeLaya(), fake_gemini(sizes), display=display, console=console)

    assert sizes == [2, 2, 1]
    assert all(r.gemini_answer == r.gold for r in run.records)
    assert run.gemini_baseline_idx == [0, 1, 2, 3, 4]
    assert [r.final_source for r in run.records] == ["laya", "laya", "gemini", "gemini", "human"]
    assert run.meta["race"]["laya_s"] >= 0 and run.meta["race"]["gemini_s"] >= 0
    if display:
        assert "faster per message" in console.file.getvalue()


def test_race_without_gemini_still_runs_laya():
    run = run_race(Config(n=len(ITEMS)), ITEMS, LABELS, FakeLaya(), None, display=False)
    assert all(r.gemini_answer is None for r in run.records)
    assert run.meta["race"]["gemini_s"] is None
    assert "no gemini_api_key" in run.meta["gemini_status"]


def test_report_prints_tables_items_and_speed(tmp_path, capsys):
    cfg = Config(n=len(ITEMS))
    run = run_race(cfg, ITEMS, LABELS, FakeLaya(), fake_gemini(), display=False)
    from laya_demo.cache import save_run

    save_run(run, tmp_path)
    laya_demo.report_main(["--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Per tier" in out and "Per item" in out and "Wall clock: Laya 5 items" in out
    assert Path(tmp_path / "reliability.png").exists()
