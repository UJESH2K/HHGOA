"""The Fraud Policy's action vocabulary, approval routing, and the two gates.

Deterministic. The LLM proposes; this module decides what the proposal is allowed to be and who
must approve it. The brief separates recommending from executing, so the model cannot also be the
authority on what is authorized.

Action names and route identifiers are verbatim from Fraud Policy v1.0 sections 1-3a. Do not
rename them - the answer files are scored on these exact strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    ALLOW_TRANSACTION = "ALLOW_TRANSACTION"
    DECLINE_TRANSACTION = "DECLINE_TRANSACTION"
    MONITOR_CARD = "MONITOR_CARD"
    MONITOR_CONNECTED_CARDS = "MONITOR_CONNECTED_CARDS"
    WARN_CUSTOMER = "WARN_CUSTOMER"
    VERIFY_WITH_CUSTOMER = "VERIFY_WITH_CUSTOMER"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    BLOCK_CARD = "BLOCK_CARD"
    BLOCK_ALL_CARDS = "BLOCK_ALL_CARDS"
    GENERATE_REPORT = "GENERATE_REPORT"
    CREATE_CASE = "CREATE_CASE"
    FILE_REPORT = "FILE_REPORT"
    ESCALATE_TO_ANALYST = "ESCALATE_TO_ANALYST"
    CLOSE_NO_FRAUD = "CLOSE_NO_FRAUD"


class Route(str, Enum):
    AUTO = "auto"
    L1 = "L1"
    L2 = "L2"


BLOCK_CARD_L2_THRESHOLD = 2500.0   # policy 2: BLOCK_CARD is L2 above this exposure
SAR_EXPOSURE_THRESHOLD = 1000.0    # policy 3a: one of the report triggers
CASE_PROBABILITY_THRESHOLD = 0.30  # policy 3a: open a case at or above this

_FIXED_ROUTES = {
    Action.DECLINE_TRANSACTION: Route.L1,
    Action.BLOCK_ALL_CARDS: Route.L2,
    Action.FILE_REPORT: Route.L2,
}


def route_for(action: Action | str, exposure_usd: float = 0.0) -> Route:
    """Approval route for an action at a given exposure. Policy section 2."""
    action = Action(action)
    if action == Action.BLOCK_CARD:
        return Route.L1 if exposure_usd <= BLOCK_CARD_L2_THRESHOLD else Route.L2
    return _FIXED_ROUTES.get(action, Route.AUTO)


def is_executable(action: Action | str, exposure_usd: float = 0.0) -> bool:
    """Only `auto` actions may be executed by the agent. L1/L2 wait for a human."""
    return route_for(action, exposure_usd) == Route.AUTO


# "An agent may recommend several actions for one case. Order them by what happens first."
ACTION_ORDER = [
    Action.DECLINE_TRANSACTION,        # the flagged authorization, right now
    Action.ALLOW_TRANSACTION,
    Action.STEP_UP_AUTH,               # gate further activity
    Action.VERIFY_WITH_CUSTOMER,       # ask the cardholder
    Action.BLOCK_CARD,                 # protective
    Action.BLOCK_ALL_CARDS,
    Action.MONITOR_CARD,
    Action.MONITOR_CONNECTED_CARDS,
    Action.CREATE_CASE,                # record
    Action.GENERATE_REPORT,
    Action.FILE_REPORT,                # regulatory
    Action.WARN_CUSTOMER,
    Action.ESCALATE_TO_ANALYST,        # handoff / closure
    Action.CLOSE_NO_FRAUD,
]
_ORDER_INDEX = {a: i for i, a in enumerate(ACTION_ORDER)}


@dataclass(frozen=True)
class Recommendation:
    action: Action
    route: Route
    reason: str

    def to_json(self) -> dict:
        return {"action": self.action.value, "route": self.route.value, "reason": self.reason}


@dataclass
class Decision:
    """The full output of the policy engine for one pass."""
    recommendations: list[Recommendation] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def to_json(self) -> list[dict]:
        return [r.to_json() for r in self.recommendations]

    @property
    def action_names(self) -> list[str]:
        return [r.action.value for r in self.recommendations]


def compose(proposed: list[tuple[Action, str]], exposure_usd: float) -> Decision:
    """Order, de-duplicate and route a set of (action, reason) pairs.

    De-duplication keeps the FIRST reason given for an action, so the rule that raised it owns
    the citation. Contradictory pairs are resolved in favour of the protective action, because a
    rule that fired for a reason should not be silently cancelled by a weaker default.
    """
    seen: dict[Action, str] = {}
    for action, reason in proposed:
        action = Action(action)
        if action not in seen:
            seen[action] = reason

    # a case cannot both stand and be declined, or be closed as clean and blocked
    if Action.DECLINE_TRANSACTION in seen:
        seen.pop(Action.ALLOW_TRANSACTION, None)
    if {Action.BLOCK_CARD, Action.BLOCK_ALL_CARDS, Action.DECLINE_TRANSACTION} & set(seen):
        seen.pop(Action.CLOSE_NO_FRAUD, None)
    if Action.BLOCK_ALL_CARDS in seen:
        seen.pop(Action.BLOCK_CARD, None)  # subsumed

    ordered = sorted(seen.items(), key=lambda kv: _ORDER_INDEX[kv[0]])
    return Decision([Recommendation(a, route_for(a, exposure_usd), r) for a, r in ordered])


def should_create_case(*, fraud_probability: float, evidence_requested: bool,
                       customer_disputes: bool) -> bool:
    """Policy 3a: a case is the internal record. Most investigations deserve one."""
    return (fraud_probability >= CASE_PROBABILITY_THRESHOLD
            or evidence_requested
            or customer_disputes)


def should_file_report(*, verdict: str, fraud_probability: float, exposure_usd: float,
                       ring_signal: bool, pattern: str) -> bool:
    """Policy 3a: a report is a regulatory filing. Most cases never need one.

    Requires confirmed-or-strongly-suspected fraud AND at least one aggravating condition.
    Closed-case history rate is 7.1% (397 of 5,565) - an agent filing on a third of the exam
    pack is mis-calibrated no matter how good the narrative reads.
    """
    strong = verdict == "fraud" or fraud_probability >= 0.70
    if not strong:
        return False
    return (exposure_usd > SAR_EXPOSURE_THRESHOLD
            or ring_signal
            or pattern == "undocumented")
