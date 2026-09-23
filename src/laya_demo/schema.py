"""Label definitions shared by every contender.

Both Laya and Gemini are rendered from the constants below, so each system sees the same
labels with the same descriptions. Change a label here and both sides change together.
"""

from __future__ import annotations

# The ten English queues in Tobi-Bueck/customer-support-tickets collapse to seven. Technical,
# IT and Product Support are near-synonyms in this dataset, and General Inquiry is too small
# and too close to Customer Service to stand alone.
QUEUE_MERGE: dict[str, str] = {
    "Technical Support": "technical",
    "IT Support": "technical",
    "Product Support": "technical",
    "Service Outages and Maintenance": "outage",
    "Customer Service": "customer_service",
    "General Inquiry": "customer_service",
    "Billing and Payments": "billing",
    "Returns and Exchanges": "returns",
    "Sales and Pre-Sales": "sales",
    "Human Resources": "hr",
}

QUEUES: dict[str, str] = {
    "technical": "a product, software, IT or hardware problem",
    "outage": "a service outage, downtime or maintenance window",
    "customer_service": "account help, general questions or feedback",
    "billing": "invoices, payments or charges",
    "returns": "returning or exchanging a product",
    "sales": "pricing, quotes or buying something new",
    "hr": "employment, hiring, payroll or staff matters",
}

TYPES: dict[str, str] = {
    "incident": "something that worked is now broken",
    "problem": "a recurring or underlying issue to investigate",
    "request": "asking for information or a service",
    "change": "asking to modify, upgrade or configure something",
}

# Ordinal, lowest first. The index is the score level Laya predicts.
PRIORITIES: dict[str, str] = {
    "low": "no time pressure, minor inconvenience",
    "medium": "needs attention soon, work is affected",
    "high": "urgent, business critical or blocking",
}

QUESTION_IDS = ("queue", "type", "priority")
LABELS: dict[str, list[str]] = {
    "queue": list(QUEUES),
    "type": list(TYPES),
    "priority": list(PRIORITIES),
}

INSTRUCTIONS: dict[str, str] = {
    "queue": "Which support team should handle the ticket in `subject` and `body`?",
    "type": "What kind of ticket is `body`?",
    "priority": "How urgent is the ticket in `body`?",
}


def ticket_state(subject: str | None, body: str) -> dict[str, str]:
    """The state object Laya reads. Gemini gets the same two fields as text."""
    return {"subject": subject or "", "body": body}


def laya_questions() -> dict[str, dict]:
    return {
        "queue": {"type": "choice", "instructions": INSTRUCTIONS["queue"], "criteria": dict(QUEUES)},
        "type": {"type": "choice", "instructions": INSTRUCTIONS["type"], "criteria": dict(TYPES)},
        "priority": {
            "type": "score",
            "instructions": INSTRUCTIONS["priority"],
            "criteria": [f"{name}: {desc}" for name, desc in PRIORITIES.items()],
        },
    }


def _label_block(title: str, labels: dict[str, str]) -> str:
    lines = "\n".join(f"- {name}: {desc}" for name, desc in labels.items())
    return f"{title}\n{lines}"


GEMINI_SYSTEM_PROMPT = "\n\n".join(
    [
        "You triage customer support tickets. For each ticket, answer three questions "
        "using only the allowed labels.",
        _label_block(f"queue - {INSTRUCTIONS['queue']}", QUEUES),
        _label_block(f"type - {INSTRUCTIONS['type']}", TYPES),
        _label_block(f"priority - {INSTRUCTIONS['priority']}", PRIORITIES),
    ]
)


def _answer_properties() -> dict:
    return {qid: {"type": "string", "enum": LABELS[qid]} for qid in QUESTION_IDS}


def gemini_single_schema() -> dict:
    return {"type": "object", "properties": _answer_properties(), "required": list(QUESTION_IDS)}


def gemini_packed_schema() -> dict:
    item = {
        "type": "object",
        "properties": {"id": {"type": "string"}, **_answer_properties()},
        "required": ["id", *QUESTION_IDS],
    }
    return {"type": "array", "items": item}


def render_ticket(ticket_id: str | None, subject: str | None, body: str) -> str:
    head = f"id: {ticket_id}\n" if ticket_id is not None else ""
    return f"{head}subject: {subject or ''}\nbody: {body}"
