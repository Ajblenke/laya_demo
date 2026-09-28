"""The model wrappers' own logic, with no weights and no network."""

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


def fake_gemini(reply):
    def generate_content(**_):
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply)

    g = GeminiSystem2.__new__(GeminiSystem2)
    g.labels, g.model, g._config = LABELS, "fake", None
    g._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return g


def test_gemini_answer_on_list():
    a = fake_gemini(" card_arrival\n").answer("where is my card")
    assert (a.answer, a.error) == ("card_arrival", None)


def test_gemini_off_list_and_errors_are_recorded_not_raised():
    assert "not in label list" in fake_gemini("something else").answer("x").error
    failed = fake_gemini(ConnectionError("offline")).answer("x")
    assert failed.answer is None and "offline" in failed.error


def test_default_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-other")
    assert system2.default_model() == "gemini-other"
