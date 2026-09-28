from laya_demo.cache import load_run, save_run
from laya_demo.pipeline import finalize, route_records
from laya_demo.records import Record, RunResult
from laya_demo.report import calibration_summary, render, tier_table, tradeoff_table
from laya_demo.routing import make_tiers


def sample_run() -> RunResult:
    records = [
        Record(0, "a", "x", "x", 0.9, 0.9, 10.0),  # local, right
        Record(1, "b", "x", "y", 0.8, 0.8, 10.0),  # local, wrong
        Record(2, "c", "x", "y", 0.6, 0.6, 10.0),  # retrieval, Gemini fixes it
        Record(3, "d", "x", "y", 0.4, 0.4, 10.0),  # larger_model, no Gemini answer
        Record(4, "e", "x", "y", 0.1, 0.1, 10.0),  # human
    ]
    route_records(records, make_tiers(), "confidence")
    records[2].gemini_answer, records[2].gemini_latency_ms = "x", 490.0
    records[0].gemini_answer, records[0].gemini_latency_ms = "x", 300.0  # baseline only
    finalize(records)
    meta = {
        "started_at": "2026-09-28T10:00:00",
        "thresholds": [0.75, 0.55, 0.35],
        "signal": "confidence",
        "device": "cpu",
        "gemini_status": "test",
    }
    return RunResult(meta, ["x", "y"], records, gemini_baseline_idx=[0, 2])


def test_tier_table_counts_and_accuracy():
    run = sample_run()
    rows = {r["tier"]: r for r in tier_table(run.records, run.tiers)}
    assert [rows[t]["count"] for t in ("local", "retrieval", "larger_model", "human")] == [2, 1, 1, 1]
    assert rows["local"]["laya_acc"] == 0.5
    assert rows["retrieval"]["laya_acc"] == 0.0 and rows["retrieval"]["final_acc"] == 1.0
    assert rows["retrieval"]["mean_latency_ms"] == 500.0
    assert rows["human"]["final_acc"] is None


def test_tradeoff_rows():
    rows = {r["system"]: r for r in tradeoff_table(sample_run())}
    assert rows["Laya alone"]["accuracy"] == 0.2
    assert rows["Gemini alone"]["n"] == 2 and rows["Gemini alone"]["accuracy"] == 1.0
    assert rows["Gemini alone"]["mean_latency_ms"] == 395.0
    # automated: 0 right, 1 wrong, 2 right (Gemini), 3 wrong (fallback); 4 goes to a human
    assert rows["Routed"]["accuracy"] == 0.5 and rows["Routed"]["coverage"] == 0.8


def test_calibration_summary_and_render():
    run = sample_run()
    cal = calibration_summary(run.records)
    assert cal["accuracy"] == 0.2 and cal["ece_confidence"] > 0
    text = render(run)
    assert "Per tier" in text and "Trade off" in text and "ECE" in text


def test_cache_round_trip(tmp_path):
    run = sample_run()
    path = save_run(run, tmp_path)
    assert path.exists() and (tmp_path / "last_run.json").exists()
    again = load_run(data_dir=tmp_path)
    assert again.to_dict() == run.to_dict()
    assert again.records[2].final_source == "gemini"
