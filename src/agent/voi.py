"""Which question is worth asking: expected value of information over evidence requests.

THE PROBLEM THIS SOLVES. The brief asks the agent to "gather additional evidence when needed
through controlled, policy-approved actions" and to "determine when to stop the investigation
once enough evidence is available to take a defensible action". The obvious implementation is a
reflex: if the probability is mid-band, ask the customer. That reflex is wrong in both
directions - it asks on cases where the answer cannot change anything (the customer already
disputed the transaction as the trigger, so R2 already applies), and it asks the cheap generic
question on cases where a different one would actually settle the matter.

THE APPROACH. An evidence request is worth making only if the action it leads to could differ
from the action taken without it. So for each request the policy allows, enumerate the possible
responses, re-run the rubric under each one, run the policy engine on the result, and measure how
much the recommended action set moves. Weight those movements by how likely each response is,
subtract the friction of asking, and rank.

    EVOI(request) = SUM over responses  P(response) * decision_distance(now, after)
    net(request)  = EVOI(request) - friction(request)

WHAT THIS BUYS BEYOND A REFLEX.

  - A recorded reason for asking, and for NOT asking: every rejected option keeps its number, so
    the case file can say "asking the cardholder would not have changed the recommendation".
  - The right question rather than the default one. A failed step-up is strong evidence about
    whether the cardholder is in control of the account; a second dispute from someone who has
    already disputed is not evidence at all.
  - A defensible stopping rule. When no request has positive net value, the investigation is
    finished in the only sense that matters: nothing available would change what we do.

HONESTY ABOUT THE RESPONSE MODEL. The organizers do not supply responses, so the probabilities
below are assumptions, not measurements. They are declared in one place, derived from the current
belief rather than hardcoded per case, and every one of them is written into the case file next to
the request it justified. The ranking is not sensitive to small changes in them - what drives it
is whether a response can move the policy engine at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from ..policy.actions import Action, Decision
from ..policy.rules import PolicyContext, evaluate
from ..scoring.rubric import RubricInput, RubricScore, score

# The three request types the policy permits. Anything else is not a controlled action.
CUSTOMER_VALIDATION = "customer_validation"
STEP_UP_AUTH = "step_up_auth"
ANALYST_INFO = "analyst_info"

FRICTION = {
    # Cost of asking, in the same units as decision distance (0-1). Not money: the cost of
    # bothering a customer, burning an authentication step, or consuming an analyst's attention.
    # Step-up is automated and near-free; a phone call to a cardholder is not; an analyst's time
    # is the scarcest of the three, and R8 exists to route cases there deliberately rather than
    # as a way of avoiding a decision.
    STEP_UP_AUTH: 0.05,
    CUSTOMER_VALIDATION: 0.12,
    ANALYST_INFO: 0.20,
}

MIN_NET_VALUE = 0.05
"""Below this, asking is not worth the friction and the investigation should conclude.

Deliberately small but non-zero: a request that shifts the recommendation on a 1-in-20 branch is
still worth making when the shift is between blocking a card and allowing a transaction, but a
request that only reshuffles monitoring actions is not.
"""

# --------------------------------------------------------------------------------------------
# Measuring how far a recommendation moved.
#
# The first version of this was a severity-weighted symmetric difference over action sets, and
# it degenerated: on a case whose only current recommendation is "verify and step up", the set
# before and the set after share nothing by construction, so every request scored a full 1.0
# and the ranking carried no information. Comparing POSTURE fixes that - what the bank would
# actually do to the card and the money - and it reads better in a case file too: "the
# recommendation moves from monitoring to blocking the card" is a sentence an analyst can check.
# --------------------------------------------------------------------------------------------

_POSTURE = {
    Action.CLOSE_NO_FRAUD: 0.0,
    Action.ALLOW_TRANSACTION: 0.05,
    Action.CREATE_CASE: 0.10,          # a record, not a posture
    Action.GENERATE_REPORT: 0.10,
    Action.MONITOR_CARD: 0.25,
    Action.WARN_CUSTOMER: 0.25,
    Action.VERIFY_WITH_CUSTOMER: 0.40,  # friction on the customer, nothing restricted yet
    Action.STEP_UP_AUTH: 0.40,
    Action.MONITOR_CONNECTED_CARDS: 0.45,
    Action.DECLINE_TRANSACTION: 0.70,   # this authorisation stopped
    Action.BLOCK_CARD: 0.85,
    Action.BLOCK_ALL_CARDS: 1.00,
}
NO_POSTURE = 0.15
"""Where an empty recommendation sits: above closing the case, below monitoring."""

_POSTURE_WEIGHT, _REPORT_WEIGHT, _ESCALATE_WEIGHT = 0.70, 0.15, 0.15


def posture(d: Decision) -> float:
    """The most restrictive thing this recommendation does, on a 0-1 ladder."""
    acts = [Action(r.action) for r in d.recommendations]
    if not acts:
        return NO_POSTURE
    return max(_POSTURE.get(a, 0.25) for a in acts)


def posture_label(d: Decision) -> str:
    """The action that sets the posture, for the case file's prose."""
    acts = [Action(r.action) for r in d.recommendations]
    if not acts:
        return "no action"
    return max(acts, key=lambda a: _POSTURE.get(a, 0.25)).value


def decision_distance(before: Decision, after: Decision) -> float:
    """How far apart two recommendations are, in 0-1.

    Three components, because these are the three things that actually differ between two
    courses of action: how restrictive the posture is, whether a regulatory report gets filed,
    and whether a human is pulled in. Identical recommendations score 0; moving from closing a
    case to blocking all cards and filing scores 1.
    """
    names_before = {r.action for r in before.recommendations}
    names_after = {r.action for r in after.recommendations}
    moved = abs(posture(after) - posture(before)) * _POSTURE_WEIGHT
    report = _REPORT_WEIGHT * ((Action.FILE_REPORT.value in names_before)
                               != (Action.FILE_REPORT.value in names_after))
    escalate = _ESCALATE_WEIGHT * ((Action.ESCALATE_TO_ANALYST.value in names_before)
                                   != (Action.ESCALATE_TO_ANALYST.value in names_after))
    return min(moved + report + escalate, 1.0)


@dataclass(frozen=True)
class Branch:
    """One possible response to one request, and where it would leave the case."""
    response: str
    probability: float
    distance: float
    probability_after: float
    verdict_after: str
    actions_after: list[str]
    posture_after: str = ""

    @property
    def weighted(self) -> float:
        return self.probability * self.distance


@dataclass
class Option:
    request_type: str
    branches: list[Branch] = field(default_factory=list)
    unavailable_reason: str = ""
    posture_now: str = ""

    @property
    def evoi(self) -> float:
        return sum(b.weighted for b in self.branches)

    @property
    def friction(self) -> float:
        return FRICTION.get(self.request_type, 0.15)

    @property
    def net_value(self) -> float:
        return self.evoi - self.friction

    @property
    def worth_asking(self) -> bool:
        return not self.unavailable_reason and self.net_value >= MIN_NET_VALUE

    def rationale(self) -> str:
        """Why this request was made, or why it was not. Goes verbatim into the case record."""
        if self.unavailable_reason:
            return f"{self.request_type}: not available - {self.unavailable_reason}"
        if not self.branches:
            return f"{self.request_type}: no response to it could change the recommendation"
        moved = [b for b in self.branches if b.distance > 0.01]
        if not moved:
            return (f"{self.request_type}: every possible answer leaves the recommendation "
                    f"unchanged, so asking would add delay without adding information")
        detail = "; ".join(
            f"if {b.response}, the recommendation moves from {self.posture_now} to "
            f"{b.posture_after} (probability {b.probability_after:.2f}, "
            f"verdict {b.verdict_after})"
            for b in moved)
        verb = "worth asking" if self.worth_asking else "not worth the friction of asking"
        return (f"{self.request_type}: expected decision change {self.evoi:.2f} against friction "
                f"{self.friction:.2f}, so {verb}. {detail}")


@dataclass
class Selection:
    """The full comparison, ranked. `best` is None when the investigation should conclude."""
    options: list[Option]
    best: Option | None

    @property
    def stop_reason(self) -> str:
        if self.best is not None:
            return ""
        considered = ", ".join(o.request_type for o in self.options)
        return ("No further evidence request would change the recommended action: every "
                f"policy-approved option ({considered}) was evaluated against its possible "
                "responses and none moved the decision by more than the cost of asking.")

    def to_json(self) -> list[dict]:
        return [{
            "request_type": o.request_type,
            "expected_decision_change": round(o.evoi, 4),
            "friction": o.friction,
            "net_value": round(o.net_value, 4),
            "selected": o is self.best,
            "rationale": o.rationale(),
            "posture_now": o.posture_now,
            "branches": [{"response": b.response, "probability": b.probability,
                          "decision_change": round(b.distance, 4),
                          "fraud_probability_after": round(b.probability_after, 3),
                          "verdict_after": b.verdict_after,
                          "posture_after": b.posture_after,
                          "actions_after": b.actions_after}
                         for b in o.branches],
        } for o in sorted(self.options, key=lambda o: o.net_value, reverse=True)]


# --------------------------------------------------------------------------------------------
# The response model. Assumptions, stated once, conditioned on the current belief.
# --------------------------------------------------------------------------------------------

def customer_validation_branches(p_fraud: float) -> dict[str, float]:
    """P(response) when we ask the cardholder whether they made the transaction.

    Conditioned on the current belief, because that is the honest structure: if the evidence
    already points at fraud, a denial is the likely answer. The floor on `confirmed` is
    deliberate - policy R7 exists because cardholders do dispute charges they made and forgot,
    and an agent that treats a dispute as proof will block subscriptions.
    """
    denied = 0.25 + 0.60 * p_fraud
    confirmed = 0.60 - 0.45 * p_fraud
    no_reply = max(0.0, 1.0 - denied - confirmed)
    return {"denied": round(denied, 3), "confirmed": round(confirmed, 3),
            "no_reply": round(no_reply, 3)}


def step_up_branches(p_fraud: float) -> dict[str, float]:
    """P(response) for a step-up authentication challenge.

    A genuine cardholder usually passes; someone using a stolen card usually cannot. This is the
    request with the highest information density in the policy's toolkit, which is why its
    friction is set lowest - it is also the one a reflex-based agent under-uses.
    """
    failed = 0.05 + 0.80 * p_fraud
    return {"failed": round(failed, 3), "passed": round(1.0 - failed, 3)}


def analyst_info_branches(i: RubricInput) -> dict[str, float]:
    """P(response) for asking an analyst or approved party for more context.

    An analyst cannot answer the question "did the cardholder make this transaction" - only the
    cardholder and the authentication system can. What an analyst can do is resolve something
    already half-visible: confirm that a shared-origin link the graph discounted is real, or
    confirm that a merchant relationship is established and the charge is recurring.

    So the branch set depends on whether there is anything for them to resolve. The first version
    of this modelled a flat "material" response that set the ring flag unconditionally, which let
    the selector conjure a fraud ring out of a case with no shared-origin evidence at all, and
    then rank that fabrication as the most valuable question to ask. Asking an analyst about a
    link that does not exist cannot confirm one.
    """
    if i.ring_signal and i.ring_volume_artefact:
        # the graph found a link and we discounted it as a volume artefact: worth confirming
        return {"confirms_link": 0.40, "confirms_known_relationship": 0.10, "nothing_new": 0.50}
    if i.trigger_type == "analyst_request":
        # an analyst raised this case, so they have context we do not
        return {"confirms_link": 0.25, "confirms_known_relationship": 0.25, "nothing_new": 0.50}
    return {"confirms_known_relationship": 0.25, "nothing_new": 0.75}


# --------------------------------------------------------------------------------------------
# Applying a hypothetical response
# --------------------------------------------------------------------------------------------

def apply_response(i: RubricInput, request_type: str, response: str) -> RubricInput:
    """The counterfactual input: what we would know if this answer came back.

    Only the fields the response actually speaks to are changed. An analyst confirming a
    connection sets the ring flag; it does not invent an amount percentile.
    """
    if request_type == CUSTOMER_VALIDATION:
        return replace(i, customer_response=response)
    if request_type == STEP_UP_AUTH:
        return replace(i, step_up_result=response)
    if request_type == ANALYST_INFO:
        if response == "confirms_link":
            # the analyst confirms a shared-origin link the graph found but could not qualify
            return replace(i, ring_signal=True, ring_volume_artefact=False,
                           ring_n_cards=max(i.ring_n_cards, 2))
        if response == "confirms_known_relationship":
            # an established merchant relationship: R7 territory, not fraud
            return replace(i, matches_recurring_pattern=True, known_email_domain=True)
        return i
    raise ValueError(f"unknown request type {request_type!r}")


def _availability(i: RubricInput, request_type: str) -> str:
    """Why a request cannot or should not be made. Empty string means it is available.

    These are not optimisations - they are the policy's own logic. Asking a customer who has
    already told you the transaction is not theirs is not evidence gathering, and re-challenging
    an authentication that already failed tells you nothing new.
    """
    if request_type == CUSTOMER_VALIDATION:
        if i.customer_response is not None:
            return f"the cardholder has already responded ({i.customer_response})"
        if i.trigger_type == "customer_report":
            return ("the trigger is the cardholder's own report, so the dispute is already on "
                    "record and R2 applies without asking again")
    if request_type == STEP_UP_AUTH:
        if i.step_up_result is not None:
            return f"step-up has already been attempted ({i.step_up_result})"
        if i.matches_recurring_pattern:
            return ("the charge matches the cardholder's own recurring pattern, and R7 says "
                    "explicitly not to escalate friction onto it")
    return ""


ContextBuilder = Callable[[RubricInput, RubricScore], PolicyContext]


def evaluate_options(
    i: RubricInput,
    to_context: ContextBuilder,
    *,
    request_types: tuple[str, ...] = (STEP_UP_AUTH, CUSTOMER_VALIDATION, ANALYST_INFO),
    already_requested: tuple[str, ...] = (),
) -> Selection:
    """Rank every policy-approved evidence request by what it would change.

    `to_context` is how a hypothetical assessment becomes a `PolicyContext` - the caller supplies
    it (see `agent/context.build_policy_context`) so that the exposure, pattern and cardholder
    facts used in the counterfactual are exactly the ones the real decision will use.
    """
    now_score = score(i)
    now_decision = evaluate(to_context(i, now_score))

    options: list[Option] = []
    for rt in request_types:
        if rt in already_requested:
            options.append(Option(rt, unavailable_reason="already requested in this case"))
            continue
        reason = _availability(i, rt)
        if reason:
            options.append(Option(rt, unavailable_reason=reason))
            continue

        if rt == ANALYST_INFO:
            priors = analyst_info_branches(i)
        elif rt == CUSTOMER_VALIDATION:
            priors = customer_validation_branches(now_score.probability)
        else:
            priors = step_up_branches(now_score.probability)

        branches = []
        for response, prob in priors.items():
            hypo = apply_response(i, rt, response)
            hypo_score = score(hypo)
            hypo_decision = evaluate(to_context(hypo, hypo_score))
            branches.append(Branch(
                response=response, probability=prob,
                distance=decision_distance(now_decision, hypo_decision),
                probability_after=hypo_score.probability,
                verdict_after=hypo_score.verdict,
                actions_after=hypo_decision.action_names,
                posture_after=posture_label(hypo_decision)))
        options.append(Option(rt, branches=branches, posture_now=posture_label(now_decision)))

    viable = [o for o in options if o.worth_asking]
    best = max(viable, key=lambda o: o.net_value) if viable else None
    return Selection(options=options, best=best)
