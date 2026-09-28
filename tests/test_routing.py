import math

import pytest

from laya_demo.pipeline import finalize, plan_gemini_calls, route_records
from laya_demo.records import Record
from laya_demo.routing import assign_tier, make_tiers, tier_bounds


def rec(idx, conf, top=None, gold="a", laya="a", latency=10.0):
    return Record(idx, f"text {idx}", gold, laya, conf, conf if top is None else top, latency)


@pytest.mark.parametrize(
    ("conf", "tier"),
    [
        (1.0, "local"),
        (0.75, "local"),
        (0.7499, "retrieval"),
        (0.55, "retrieval"),
        (0.5499, "larger_model"),
        (0.35, "larger_model"),
        (0.3499, "human"),
        (0.0, "human"),
        (math.nan, "human"),
    ],
)
def test_paper_thresholds_put_boundaries_in_the_higher_tier(conf, tier):
    assert assign_tier(conf, make_tiers()).name == tier


def test_thresholds_are_parameters():
    tiers = make_tiers(0.9, 0.6, 0.2)
    assert assign_tier(0.8, tiers).name == "retrieval"
    assert assign_tier(0.1, tiers).name == "human"
    assert tier_bounds(tiers) == {
        "local": ">= 0.90",
        "retrieval": "0.60 to 0.90",
        "larger_model": "0.20 to 0.60",
        "human": "< 0.20",
    }


@pytest.mark.parametrize("bad", [(0.5, 0.6, 0.2), (0.9, 0.6, 0.7), (1.2, 0.6, 0.3), (0.9, 0.6, -0.1)])
def test_unordered_or_out_of_range_thresholds_are_rejected(bad):
    with pytest.raises(ValueError):
        make_tiers(*bad)


def test_route_records_uses_the_chosen_signal():
    records = [rec(0, conf=0.9, top=0.4)]
    route_records(records, make_tiers(), "confidence")
    assert (records[0].tier, records[0].action) == ("local", "accept")
    route_records(records, make_tiers(), "top_prob")
    assert (records[0].tier, records[0].action) == ("larger_model", "escalate")


def test_plan_puts_escalations_first_and_respects_the_cap():
    records = [rec(i, conf=0.9) for i in range(20)]
    for i in (3, 7, 11):
        records[i].action = "escalate"
    calls, baseline = plan_gemini_calls(records, baseline_n=10, cap=5, seed=0)
    assert calls[:3] == [3, 7, 11]
    assert len(calls) == 5
    assert set(baseline) <= set(calls)


def test_plan_baseline_is_a_uniform_sample_not_the_escalated_items():
    records = [rec(i, conf=0.9) for i in range(50)]
    for r in records[:5]:
        r.action = "escalate"
    calls, baseline = plan_gemini_calls(records, baseline_n=20, cap=100, seed=1)
    assert len(baseline) == 20
    assert len(calls) == len(set(calls))
    assert set(range(5)) <= set(calls)


def test_zero_cap_calls_nothing():
    records = [rec(0, conf=0.1)]
    records[0].action = "escalate"
    assert plan_gemini_calls(records, baseline_n=5, cap=0, seed=0) == ([], [])


def test_finalize_by_action():
    accept, escalated, fallback, human = (rec(i, conf=0.5, laya="x") for i in range(4))
    accept.action = "accept"
    escalated.action = fallback.action = "escalate"
    escalated.gemini_answer, escalated.gemini_latency_ms = "a", 500.0
    human.action = "human"
    finalize([accept, escalated, fallback, human])
    assert (accept.final_answer, accept.final_source) == ("x", "laya")
    assert (escalated.final_answer, escalated.final_source) == ("a", "gemini")
    assert (fallback.final_answer, fallback.final_source) == ("x", "laya_fallback")
    assert (human.final_answer, human.final_source, human.final_correct) == (None, "human", None)
    assert escalated.routed_latency_ms == 510.0
    assert accept.routed_latency_ms == 10.0


def test_baseline_gemini_call_on_an_accepted_item_does_not_count_as_routed_latency():
    r = rec(0, conf=0.9)
    r.action = "accept"
    r.gemini_answer, r.gemini_latency_ms = "a", 800.0
    assert r.routed_latency_ms == 10.0
