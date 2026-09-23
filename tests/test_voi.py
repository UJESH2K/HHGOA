"""Value-of-information tests. No dataset, no credentials.

The behaviours pinned here are the ones that make the selector better than a reflex, and each of
them is a claim we will make in the write-up:

  - it declines to ask a question whose answer cannot change the recommendation
  - it declines to ask a cardholder who has already told us the transaction is not theirs
  - it prefers the cheap, high-information challenge over the expensive, low-information one
  - it cannot invent evidence in a counterfactual that the case does not support
"""
from __future__ import annotations

import pytest

from src.agent.context import build_policy_context
from src.agent.voi import (ANALYST_INFO, CUSTOMER_VALIDATION, MIN_NET_VALUE, NO_POSTURE,
                           STEP_UP_AUTH, analyst_info_branches, apply_response,
                           customer_validation_branches, decision_distance, evaluate_options,
                           posture, posture_label, step_up_branches)
from src.policy.actions import Action, Decision, Recommendation, Route
from src.policy.rules import PolicyContext, evaluate
from src.scoring.rubric import RubricInput


def ctx_for(exposure: float, **kw):
    def build(i, s):
        return build_policy_context(i, s, exposure_usd=exposure, **kw)
    return build


def opt(sel, request_type):
    return next(o for o in sel.options if o.request_type == request_type)


# --- decision distance ----------------------------------------------------------------------

def _decision(*actions: Action) -> Decision:
    return Decision([Recommendation(a, Route.AUTO, "test") for a in actions])


def test_identical_recommendations_are_zero_apart():
    d = _decision(Action.MONITOR_CARD, Action.CREATE_CASE)
    assert decision_distance(d, d) == 0.0


def test_closing_a_case_versus_blocking_every_card_is_the_full_range():
    a = _decision(Action.CLOSE_NO_FRAUD)
    b = _decision(Action.BLOCK_ALL_CARDS, Action.FILE_REPORT, Action.ESCALATE_TO_ANALYST)
    assert decision_distance(a, b) == pytest.approx(1.0)


def test_distance_is_symmetric():
    a = _decision(Action.MONITOR_CARD)
    b = _decision(Action.BLOCK_CARD, Action.FILE_REPORT)
    assert decision_distance(a, b) == decision_distance(b, a)


def test_filing_a_report_registers_even_at_the_same_posture():
    a = _decision(Action.BLOCK_CARD)
    b = _decision(Action.BLOCK_CARD, Action.FILE_REPORT)
    assert 0 < decision_distance(a, b) < 0.5


def test_an_empty_recommendation_has_a_defined_posture():
    """The degenerate case that broke the first metric: nothing recommended yet."""
    assert posture(_decision()) == NO_POSTURE
    assert posture_label(_decision()) == "no action"


def test_posture_is_the_most_restrictive_action():
    d = _decision(Action.CREATE_CASE, Action.BLOCK_CARD, Action.MONITOR_CARD)
    assert posture_label(d) == Action.BLOCK_CARD.value


# --- the response model ---------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.0, 0.15, 0.5, 0.85, 1.0])
def test_response_priors_are_distributions(p):
    for branches in (customer_validation_branches(p), step_up_branches(p)):
        assert all(v >= 0 for v in branches.values())
        assert sum(branches.values()) == pytest.approx(1.0, abs=1e-6)


def test_a_denial_gets_likelier_as_the_evidence_gets_worse():
    assert (customer_validation_branches(0.9)["denied"]
            > customer_validation_branches(0.1)["denied"])


def test_a_step_up_failure_gets_likelier_as_the_evidence_gets_worse():
    assert step_up_branches(0.9)["failed"] > step_up_branches(0.1)["failed"]


def test_confirmation_never_drops_to_zero():
    """R7 exists because cardholders dispute charges they made and forgot."""
    assert customer_validation_branches(1.0)["confirmed"] > 0.05


# --- counterfactuals may not invent evidence ------------------------------------------------

def test_an_analyst_cannot_confirm_a_link_that_was_never_found():
    """The bug this guards: `analyst_info` set the ring flag unconditionally, so the selector
    conjured a fraud ring on a case with no shared-origin evidence and then ranked asking about
    that fabrication as the most valuable question available."""
    plain = RubricInput(amt_pctile_prior=0.97, n_prior_txns=100, amount_usd=482.0)
    assert "confirms_link" not in analyst_info_branches(plain)

    discounted = RubricInput(ring_signal=True, ring_volume_artefact=True, ring_n_cards=9)
    assert "confirms_link" in analyst_info_branches(discounted)


def test_analyst_info_branches_are_a_distribution():
    for i in (RubricInput(), RubricInput(trigger_type="analyst_request"),
              RubricInput(ring_signal=True, ring_volume_artefact=True)):
        assert sum(analyst_info_branches(i).values()) == pytest.approx(1.0, abs=1e-6)


def test_apply_response_only_touches_what_the_answer_speaks_to():
    i = RubricInput(amt_pctile_prior=0.9, n_prior_txns=50, amount_usd=100.0)
    after = apply_response(i, CUSTOMER_VALIDATION, "denied")
    assert after.customer_response == "denied"
    assert after.amt_pctile_prior == i.amt_pctile_prior
    assert after.ring_signal == i.ring_signal


def test_unknown_request_types_are_rejected():
    with pytest.raises(ValueError):
        apply_response(RubricInput(), "bribe_the_merchant", "yes")


# --- availability ---------------------------------------------------------------------------

def test_a_cardholder_who_already_disputed_is_not_asked_again():
    """HHG-006 shape. R2 applies on the trigger; asking again is delay, not evidence."""
    sel = evaluate_options(
        RubricInput(trigger_type="customer_report", amt_pctile_prior=0.97, n_prior_txns=101,
                    amount_usd=482.12),
        ctx_for(482.12))
    o = opt(sel, CUSTOMER_VALIDATION)
    assert o.unavailable_reason
    assert "R2" in o.unavailable_reason
    assert not o.worth_asking


def test_step_up_is_not_pushed_onto_a_recurring_charge():
    """R7 says explicitly not to escalate friction onto the cardholder's own subscription."""
    sel = evaluate_options(
        RubricInput(trigger_type="customer_report", matches_recurring_pattern=True,
                    amt_pctile_prior=0.18, n_prior_txns=5860, amount_usd=39.08),
        ctx_for(39.08))
    assert "R7" in opt(sel, STEP_UP_AUTH).unavailable_reason
    assert sel.best is None
    assert sel.stop_reason


def test_a_request_is_never_made_twice():
    i = RubricInput(amt_pctile_prior=1.0, n_prior_txns=35, amount_usd=292.36)
    sel = evaluate_options(i, ctx_for(292.36), already_requested=(CUSTOMER_VALIDATION,))
    assert "already requested" in opt(sel, CUSTOMER_VALIDATION).unavailable_reason
    assert sel.best is None or sel.best.request_type != CUSTOMER_VALIDATION


def test_an_answered_question_is_not_reasked():
    sel = evaluate_options(RubricInput(customer_response="no_reply"), ctx_for(100.0))
    assert "already responded" in opt(sel, CUSTOMER_VALIDATION).unavailable_reason


# --- the selection itself -------------------------------------------------------------------

def test_a_single_signal_case_asks_for_the_second_signal():
    """HHG-002 shape: top-percentile amount and nothing else, so one answer settles it.

    Both the challenge and the phone call are informative here, and the selector takes the
    cheaper one - which is the point of pricing friction at all. What matters is that it asks:
    a case sitting at 0.76 on a single family must not be acted on as though it were decided.
    """
    sel = evaluate_options(
        RubricInput(amt_pctile_prior=1.0, n_prior_txns=35, amount_usd=292.36,
                    trigger_type="risk_score", risk_score=0.79),
        ctx_for(292.36))
    assert sel.best is not None
    assert sel.best.net_value >= MIN_NET_VALUE
    assert sel.best.request_type in (STEP_UP_AUTH, CUSTOMER_VALIDATION)
    # The chosen question wins on net value, not on raw information. Both were priced; only one
    # has to clear the friction, and which one depends on what the policy already recommends -
    # once R1 has put a verification step in the initial actions, a step-up adds less.
    assert sel.best.net_value >= max(o.net_value for o in sel.options if o is not sel.best)
    assert all(o.branches or o.unavailable_reason for o in sel.options), "every option priced"


def test_the_cheaper_question_wins_a_tie_on_information():
    """Friction is priced so an automated challenge beats a phone call at equal value."""
    i = RubricInput(amt_pctile_prior=1.0, n_prior_txns=35, amount_usd=292.36,
                    trigger_type="risk_score", risk_score=0.79)
    sel = evaluate_options(i, ctx_for(292.36))
    step_up, customer = opt(sel, STEP_UP_AUTH), opt(sel, CUSTOMER_VALIDATION)
    if abs(step_up.evoi - customer.evoi) < 0.10:
        assert step_up.net_value > customer.net_value


def test_a_likely_false_alarm_prefers_the_cheap_challenge():
    """HHG-013 shape: high score, 15th-percentile amount. Step up before bothering anyone."""
    sel = evaluate_options(
        RubricInput(amt_pctile_prior=0.147, n_prior_txns=1431, amount_usd=35.66,
                    trigger_type="risk_score", risk_score=0.76, device_state="New"),
        ctx_for(35.66))
    assert sel.best is not None
    assert sel.best.request_type == STEP_UP_AUTH
    assert sel.best.net_value > opt(sel, CUSTOMER_VALIDATION).net_value


def test_a_resolved_case_stops_asking():
    """Two independent signals and a fraud verdict: no answer would change the action."""
    sel = evaluate_options(
        RubricInput(trigger_type="customer_report", amt_pctile_prior=0.97, n_prior_txns=101,
                    amount_usd=482.12, device_state="New"),
        ctx_for(482.12))
    assert sel.best is None
    assert "would change the recommended action" in sel.stop_reason


def test_every_option_carries_a_rationale_even_when_rejected():
    sel = evaluate_options(
        RubricInput(trigger_type="customer_report", amt_pctile_prior=0.97, n_prior_txns=101,
                    amount_usd=482.12),
        ctx_for(482.12))
    for o in sel.options:
        assert o.rationale().startswith(o.request_type)
        assert len(o.rationale()) > 40


def test_to_json_is_ranked_and_marks_the_selection():
    sel = evaluate_options(
        RubricInput(amt_pctile_prior=1.0, n_prior_txns=35, amount_usd=292.36), ctx_for(292.36))
    rows = sel.to_json()
    assert [r["net_value"] for r in rows] == sorted((r["net_value"] for r in rows), reverse=True)
    assert sum(1 for r in rows if r["selected"]) == (1 if sel.best else 0)
    assert all({"request_type", "expected_decision_change", "friction", "net_value", "selected",
                "rationale", "posture_now", "branches"} == set(r) for r in rows)


def test_branch_probabilities_and_distances_are_in_range():
    sel = evaluate_options(
        RubricInput(amt_pctile_prior=0.9, n_prior_txns=200, amount_usd=300.0, risk_score=0.6),
        ctx_for(300.0))
    for o in sel.options:
        for b in o.branches:
            assert 0.0 <= b.probability <= 1.0
            assert 0.0 <= b.distance <= 1.0
            assert 0.0 <= b.probability_after <= 1.0
            assert b.posture_after


def test_the_counterfactual_uses_the_same_context_as_the_real_decision():
    """A selector optimising for a decision the agent would not then take is worse than none."""
    i = RubricInput(amt_pctile_prior=1.0, n_prior_txns=35, amount_usd=292.36)
    build = ctx_for(292.36, pattern="account_takeover", customer_n_cards=2)
    sel = evaluate_options(i, build)
    denied = next(b for b in opt(sel, CUSTOMER_VALIDATION).branches if b.response == "denied")

    from src.scoring.rubric import score
    hypo = apply_response(i, CUSTOMER_VALIDATION, "denied")
    expected = evaluate(build(hypo, score(hypo))).action_names
    assert denied.actions_after == expected


def test_exposure_changes_which_question_is_worth_asking():
    """R8 escalates above $500, so the same evidence is worth more on a bigger exposure."""
    i = RubricInput(amt_pctile_prior=0.92, n_prior_txns=200, amount_usd=50.0, risk_score=0.6)
    small = evaluate_options(i, ctx_for(50.0))
    large = evaluate_options(i, ctx_for(5000.0))
    assert [o.net_value for o in small.options] != [o.net_value for o in large.options]
