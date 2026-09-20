"""Fraud Policy v1.0 rules R1-R10 as predicates, plus the stopping rule.

Each rule is a pure function of a PolicyContext returning (applies, actions, citation). The
citation string lands verbatim in the answer file's `reason` field, so rule numbers are never
hand-typed by the model - policy section 7 requires every recommendation to cite its rule.

The LLM's job is to fill the PolicyContext honestly. This module's job is to turn that context
into actions the same way every time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .actions import (Action, Decision, compose, should_create_case,
                      should_file_report)

# policy section 6
STOP_HIGH = 0.85
STOP_LOW = 0.15
ESCALATE_EXPOSURE = 500.0   # R4 and R8
R1_PROBABILITY = 0.70       # R1: below this on a single signal, verify before blocking


@dataclass
class PolicyContext:
    """Everything the rules need. Filled by the agent, never by the rules themselves."""
    fraud_probability: float = 0.0
    verdict: str = "uncertain"                 # fraud | legitimate | uncertain
    pattern: str = "none"
    exposure_usd: float = 0.0

    trigger_type: str = "risk_score"           # risk_score | customer_report | analyst_request
    customer_response: str | None = None       # denied | confirmed | no_reply | None
    evidence_requested: bool = False

    n_independent_signals: int = 0
    evidence_conflicts: bool = False

    ring_signal: bool = False                  # shared device / region / another card's fraud
    ring_element: str = ""                     # R6 requires naming it
    connected_card_ids: list[str] = field(default_factory=list)

    card_testing: bool = False
    cleared_purchase_over_100: bool = False

    matches_recurring_pattern: bool = False    # R7: same merchant, same amount, monthly
    pending_authorization: bool = False

    customer_n_cards: int = 1
    n_cards_confirmed_fraud: int = 0
    credentials_compromised: bool = False

    @property
    def customer_disputes(self) -> bool:
        return self.trigger_type == "customer_report" or self.customer_response == "denied"

    @property
    def single_signal(self) -> bool:
        return self.n_independent_signals <= 1


Outcome = tuple[bool, list[tuple[Action, str]], str]


def r1_verify_before_block(c: PolicyContext) -> Outcome:
    """Verify before you block on a weak signal."""
    applies = c.single_signal and c.fraud_probability < R1_PROBABILITY
    if not applies:
        return False, [], ""
    cite = (f"R1: single signal at probability {c.fraud_probability:.2f} (<{R1_PROBABILITY}); "
            "verify before any block")
    return True, [(Action.VERIFY_WITH_CUSTOMER, cite), (Action.STEP_UP_AUTH, cite)], cite


def r2_customer_denies(c: PolicyContext) -> Outcome:
    if c.customer_response != "denied":
        return False, [], ""
    cite = "R2: customer denies the transaction"
    acts = [(Action.BLOCK_CARD, cite), (Action.CREATE_CASE, cite)]
    if c.exposure_usd > 1000 or c.ring_signal:
        connects_to = c.ring_element or "another card's fraud"
        why = ("exposure exceeds $1,000" if c.exposure_usd > 1000
               else f"case connects to {connects_to}")
        acts.append((Action.FILE_REPORT, f"R2: {why}"))
    return True, acts, cite


def r3_customer_confirms(c: PolicyContext) -> Outcome:
    if c.customer_response != "confirmed":
        return False, [], ""
    cite = "R3: customer confirms the transaction"
    return True, [(Action.CLOSE_NO_FRAUD, cite)], cite


def r4_no_reply(c: PolicyContext) -> Outcome:
    if c.customer_response != "no_reply":
        return False, [], ""
    cite = "R4: no customer reply within 24 hours"
    acts = [(Action.MONITOR_CARD, cite)]
    if c.pending_authorization:
        acts.append((Action.DECLINE_TRANSACTION, f"{cite}; pending authorization"))
    if c.exposure_usd > ESCALATE_EXPOSURE:
        acts.append((Action.ESCALATE_TO_ANALYST, f"{cite}; exposure ${c.exposure_usd:,.2f} > $500"))
    return True, acts, cite


def r5_card_testing(c: PolicyContext) -> Outcome:
    if not c.card_testing:
        return False, [], ""
    cite = "R5: card-testing sequence observed"
    acts = [(Action.DECLINE_TRANSACTION, cite), (Action.STEP_UP_AUTH, cite)]
    if c.cleared_purchase_over_100:
        acts.append((Action.BLOCK_CARD, f"{cite}; a purchase over $100 has already cleared"))
    return True, acts, cite


def r6_shared_origin(c: PolicyContext) -> Outcome:
    if not c.ring_signal:
        return False, [], ""
    element = c.ring_element or "a shared origin"
    cite = f"R6: several cards show fraud from {element}"
    return True, [(Action.CREATE_CASE, cite),
                  (Action.FILE_REPORT, cite),
                  (Action.MONITOR_CONNECTED_CARDS, f"{cite}; monitoring every card that shares it")], cite


def r7_disputed_but_legitimate(c: PolicyContext) -> Outcome:
    """Customer disputes a charge that matches their own recurring pattern. Do not block."""
    applies = c.customer_disputes and c.matches_recurring_pattern
    if not applies:
        return False, [], ""
    cite = "R7: disputed charge matches the cardholder's own recurring pattern; do not block"
    return True, [(Action.CREATE_CASE, cite),
                  (Action.VERIFY_WITH_CUSTOMER, cite),
                  (Action.WARN_CUSTOMER, cite)], cite


def r8_escalate_when_uncertain(c: PolicyContext) -> Outcome:
    applies = ((c.verdict == "uncertain" and c.exposure_usd > ESCALATE_EXPOSURE)
               or c.evidence_conflicts)
    if not applies:
        return False, [], ""
    why = ("evidence conflicts" if c.evidence_conflicts
           else f"verdict uncertain with exposure ${c.exposure_usd:,.2f} > $500")
    cite = f"R8: {why}"
    return True, [(Action.ESCALATE_TO_ANALYST, cite)], cite


def r9_undocumented(c: PolicyContext) -> Outcome:
    if c.pattern != "undocumented":
        return False, [], ""
    cite = "R9: coordinated or repeated abuse fitting none of the known patterns"
    return True, [(Action.CREATE_CASE, cite),
                  (Action.FILE_REPORT, cite),
                  (Action.ESCALATE_TO_ANALYST, cite)], cite


def r10_block_all_cards(c: PolicyContext) -> Outcome:
    """A permission, not a trigger: says when BLOCK_ALL_CARDS is *allowed*."""
    allowed = c.n_cards_confirmed_fraud >= 2 or c.credentials_compromised
    if not allowed:
        return False, [], ""
    why = ("credentials confirmed compromised" if c.credentials_compromised
           else f"{c.n_cards_confirmed_fraud} of the customer's cards show confirmed fraud")
    cite = f"R10: {why}"
    return True, [(Action.BLOCK_ALL_CARDS, cite)], cite


RULES = [r1_verify_before_block, r2_customer_denies, r3_customer_confirms, r4_no_reply,
         r5_card_testing, r6_shared_origin, r7_disputed_but_legitimate,
         r8_escalate_when_uncertain, r9_undocumented, r10_block_all_cards]


def evaluate(c: PolicyContext) -> Decision:
    """Run every rule, compose the surviving actions, route them.

    R7 is a brake: when a disputed charge matches the cardholder's own recurring pattern, the
    policy says explicitly not to block, so it suppresses R2's block rather than sitting
    alongside it.
    """
    proposed: list[tuple[Action, str]] = []
    citations: list[str] = []
    fired = {}

    for rule in RULES:
        applies, actions, cite = rule(c)
        if applies:
            fired[rule.__name__] = cite
            proposed.extend(actions)
            if cite:
                citations.append(cite)

    if "r7_disputed_but_legitimate" in fired:
        proposed = [(a, r) for a, r in proposed
                    if a not in (Action.BLOCK_CARD, Action.BLOCK_ALL_CARDS,
                                 Action.DECLINE_TRANSACTION)]

    # R10 is a permission; drop BLOCK_ALL_CARDS if nothing else called for a block
    if (Action.BLOCK_ALL_CARDS in [a for a, _ in proposed]
            and not any(a == Action.BLOCK_CARD for a, _ in proposed)
            and c.verdict != "fraud"):
        proposed = [(a, r) for a, r in proposed if a != Action.BLOCK_ALL_CARDS]

    # policy 3a gates, independent of any single rule
    if should_create_case(fraud_probability=c.fraud_probability,
                          evidence_requested=c.evidence_requested,
                          customer_disputes=c.customer_disputes):
        proposed.append((Action.CREATE_CASE, "Policy 3a: case opened for the internal record"))

    files_report = should_file_report(verdict=c.verdict, fraud_probability=c.fraud_probability,
                                      exposure_usd=c.exposure_usd, ring_signal=c.ring_signal,
                                      pattern=c.pattern)
    if not files_report:
        proposed = [(a, r) for a, r in proposed if a != Action.FILE_REPORT]

    if c.verdict == "legitimate" and not any(
            a in (Action.BLOCK_CARD, Action.DECLINE_TRANSACTION) for a, _ in proposed):
        proposed.append((Action.CLOSE_NO_FRAUD, "Verdict legitimate; no fraud found"))

    if not proposed:
        proposed.append((Action.MONITOR_CARD,
                         "No rule fired; monitoring while the picture is incomplete"))

    decision = compose(proposed, c.exposure_usd)
    decision.citations = citations
    return decision


def should_stop(c: PolicyContext, loop_count: int, max_loops: int = 3) -> tuple[bool, str]:
    """Policy section 6. Stopping too early creates risk; stopping too late wastes time."""
    if c.customer_response in ("denied", "confirmed"):
        return True, f"Customer {c.customer_response} the transaction, which settles the question."
    if loop_count >= max_loops:
        return True, (f"Evidence loop cap ({max_loops}) reached without resolution; "
                      "further steps are unlikely to change the decision.")
    confident = c.fraud_probability >= STOP_HIGH or c.fraud_probability <= STOP_LOW
    if confident and c.n_independent_signals >= 2:
        return True, (f"Probability {c.fraud_probability:.2f} is outside the "
                      f"{STOP_LOW}-{STOP_HIGH} band on {c.n_independent_signals} independent "
                      "pieces of evidence.")
    if confident:
        return False, ("Probability is decisive but rests on a single signal; policy section 6 "
                       "requires two independent pieces of evidence.")
    return False, "Probability is mid-band; more evidence could still change the decision."
