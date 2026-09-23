"""The single place a rubric assessment becomes a `PolicyContext`.

Two consumers need this mapping and they must not disagree: the Take Action node, which turns the
current assessment into recommended actions, and the value-of-information selector, which turns
*hypothetical* assessments into hypothetical actions in order to work out which question is worth
asking. If the two built their contexts differently, the agent would be optimising for a decision
it would not then take.

Everything the rules need that the rubric cannot know - exposure, pattern, how many cards the
customer holds, whether an authorisation is still pending - arrives as an explicit argument. The
rules are pure functions of this context, so anything omitted here is invisible to policy, which
is why nothing is defaulted silently to a value that would change an action.
"""
from __future__ import annotations

from collections.abc import Sequence

from ..policy.rules import PolicyContext
from ..scoring.patterns import classify
from ..scoring.rubric import RubricInput, RubricScore


def build_policy_context(
    i: RubricInput,
    s: RubricScore,
    *,
    exposure_usd: float,
    pattern: str | None = None,
    ring_element: str = "",
    connected_card_ids: Sequence[str] = (),
    customer_n_cards: int = 1,
    n_cards_confirmed_fraud: int = 0,
    credentials_compromised: bool = False,
    pending_authorization: bool = False,
    evidence_requested: bool = False,
) -> PolicyContext:
    """Map an assessment onto the policy engine's input.

    Note what is DERIVED rather than passed:

      - `verdict` and `fraud_probability` come from the rubric, including its independence guard,
        so the policy engine can never act on a confidence the evidence does not support.
      - `n_independent_signals` is the rubric's family count, which is what R1 and the stopping
        rule mean by "a single signal".
      - `evidence_conflicts` is the rubric's own conflict detection, which is what R8 needs.
      - `customer_disputes` is left to `PolicyContext` to derive from the trigger and response,
        because that is where R2 and R7 read it from.
      - `pattern` is classified from the same observations unless the caller pins it. This
        matters inside the value-of-information selector: `should_file_report` and R9 both fire
        on the pattern, so an analyst confirming a shared-origin link can turn a
        card-not-present case into an undocumented one and pull a regulatory filing with it.
        A counterfactual that held the pattern fixed would price that branch wrongly.
    """
    return PolicyContext(
        fraud_probability=s.probability,
        verdict=s.verdict,
        pattern=classify(i, s).pattern if pattern is None else pattern,
        exposure_usd=exposure_usd,
        trigger_type=i.trigger_type,
        customer_response=i.customer_response,
        evidence_requested=evidence_requested,
        n_independent_signals=s.n_independent_signals,
        evidence_conflicts=s.conflicting,
        ring_signal=i.ring_signal,
        ring_element=ring_element,
        ring_strength=i.ring_strength,
        connected_card_ids=list(connected_card_ids),
        card_testing=i.card_testing,
        cleared_purchase_over_100=i.card_testing_escalated,
        matches_recurring_pattern=i.matches_recurring_pattern,
        pending_authorization=pending_authorization,
        customer_n_cards=customer_n_cards,
        n_cards_confirmed_fraud=n_cards_confirmed_fraud,
        credentials_compromised=credentials_compromised,
    )
