"""The model wrappers' own logic, with no weights and no network."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from laya_demo import system2
from laya_demo.system1 import QUESTION_ID, build_question, parse_answer
from laya_demo.system2 import GeminiSystem2, load_api_key

LABELS = ["card_arrival", "reverted_card_payment?", "Refund_not_showing_up"]


def test_question_uses_readable_labels():
    q = build_question(LABELS)[QUESTION_ID]
    assert q["type"] == "choice"
    assert q["criteria"] == ["card arrival", "reverted card payment", "Refund not showing up"]


def test_parse_answer_maps_back_and_separates_confidence_from_top_prob():
    payload = {
        "choice": "reverted card payment",
        "probabilities": {"card arrival": 0.2, "reverted card payment": 0.7, "Refund not showing up": 0.1},
        "confidence": 0.27,
    }
    assert parse_answer(payload, LABELS) == ("reverted_card_payment?", 0.27, 0.7)


def test_parse_answer_rejects_unknown_option():
    with pytest.raises(ValueError):
        parse_answer({"choice": "nope", "probabilities": {"nope": 1.0}, "confidence": 1.0}, LABELS)


def test_key_from_dotenv_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("gemini_api_key", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_api_key() is None
    (tmp_path / ".env").write_text("gemini_api_key=\n")
    assert load_api_key() is None
    (tmp_path / ".env").write_text("gemini_api_key=placeholder-for-test\n")
    assert load_api_key() == "placeholder-for-test"


def fake_gemini(*replies, fallbacks=()):
    """A Gemini stand in that returns (or raises) each reply in turn, and records its prompts and sleeps."""
    queue = list(replies)
    prompts, sleeps = [], []

    def generate_content(contents, model, **_):
        prompts.append(contents)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)

    g = GeminiSystem2.__new__(GeminiSystem2)
    g.labels, g.model, g._config = LABELS, "fake", None
    g.fallbacks, g.answered_by, g._exhausted = list(fallbacks), Counter(), set()
    g._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    g._sleep = sleeps.append
    return g, prompts, sleeps


def reply(*pairs):
    return json.dumps([{"id": i, "label": label} for i, label in pairs])


def test_gemini_answer_on_list():
    g, _, _ = fake_gemini(reply((0, "card_arrival")))
    a = g.answer("where is my card")
    assert (a.answer, a.error) == ("card_arrival", None)


def test_gemini_off_list_and_errors_are_recorded_not_raised():
    g, _, _ = fake_gemini(reply((0, "something else")))
    assert "not in label list" in g.answer("x").error
    g, _, _ = fake_gemini(ConnectionError("offline"))
    failed = g.answer("x")
    assert failed.answer is None and "offline" in failed.error


def test_batches_are_one_request_each_and_map_back_by_id():
    g, prompts, _ = fake_gemini(
        reply((1, "Refund_not_showing_up"), (0, "card_arrival")),
        reply((0, "reverted_card_payment?")),
    )
    answers = g.answer_many(["a", "b", "c"], batch_size=2)
    assert [a.answer for a in answers] == ["card_arrival", "Refund_not_showing_up", "reverted_card_payment?"]
    assert prompts == ["[0] a\n[1] b", "[0] c"]


def test_items_missing_from_a_batch_reply_are_errors():
    g, _, _ = fake_gemini(reply((0, "card_arrival")))
    first, second = g.answer_batch(["a", "b"])
    assert first.answer == "card_arrival"
    assert second.answer is None and "missing" in second.error


def test_rate_limit_waits_the_suggested_delay_then_retries():
    limited = RuntimeError("429 RESOURCE_EXHAUSTED. {'retryDelay': '9s'}")
    g, _, sleeps = fake_gemini(limited, reply((0, "card_arrival")))
    assert g.answer("x").answer == "card_arrival"
    assert sleeps == [10.0]


def test_rate_limit_gives_up_after_the_retries():
    limited = RuntimeError("429 RESOURCE_EXHAUSTED")
    g, _, sleeps = fake_gemini(*[limited] * (system2.RATE_LIMIT_RETRIES + 1))
    assert "429" in g.answer("x").error
    assert len(sleeps) == system2.RATE_LIMIT_RETRIES


def test_an_overloaded_model_falls_back_to_the_next_one():
    busy = RuntimeError("503 UNAVAILABLE")
    tries = system2.RATE_LIMIT_RETRIES + 1
    g, _, _ = fake_gemini(*[busy] * tries, reply((0, "card_arrival")), fallbacks=["backup"])
    assert g.answer("x").answer == "card_arrival"
    assert g.answered_by == Counter({"backup": 1})


def test_an_exhausted_model_is_skipped_for_later_batches():
    busy = RuntimeError("503 UNAVAILABLE")
    tries = system2.RATE_LIMIT_RETRIES + 1
    g, prompts, _ = fake_gemini(
        *[busy] * tries, reply((0, "card_arrival")), reply((0, "card_arrival")), fallbacks=["backup"]
    )
    g.answer_many(["a", "b"], batch_size=1)
    assert len(prompts) == tries + 2
    assert g.answered_by == Counter({"backup": 2})


def test_a_rate_limit_with_a_fallback_left_moves_on_without_waiting():
    limited = RuntimeError("429 RESOURCE_EXHAUSTED. {'retryDelay': '45s'}")
    g, _, sleeps = fake_gemini(limited, reply((0, "card_arrival")), fallbacks=["backup"])
    assert g.answer("x").answer == "card_arrival"
    assert sleeps == []


def test_a_long_suggested_delay_is_capped():
    limited = RuntimeError("429 RESOURCE_EXHAUSTED. {'retryDelay': '45s'}")
    g, _, sleeps = fake_gemini(limited, reply((0, "card_arrival")))
    g.answer("x")
    assert sleeps == [system2.MAX_WAIT_S + 1]


def test_a_non_transient_error_does_not_fall_back():
    g, prompts, _ = fake_gemini(ValueError("bad request"), fallbacks=["backup"])
    assert "bad request" in g.answer("x").error
    assert len(prompts) == 1


def test_high_demand_is_retried_after_a_short_wait():
    busy = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
    g, _, sleeps = fake_gemini(busy, reply((0, "card_arrival")))
    assert g.answer("x").answer == "card_arrival"
    assert sleeps == [5.0]


def test_other_errors_are_not_retried():
    g, _, sleeps = fake_gemini(ConnectionError("offline"))
    assert "offline" in g.answer("x").error
    assert sleeps == []


def test_default_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-other")
    assert system2.default_model() == "gemini-other"
