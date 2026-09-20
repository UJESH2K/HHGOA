"""Policy engine tests. Every case here traces to a line in Fraud Policy v1.0."""
from __future__ import annotations

import pytest

from src.policy.actions import (Action, Route, is_executable, route_for,
                                should_create_case, should_file_report)
from src.policy.rules import PolicyContext, evaluate, should_stop


# --- section 2: approval routing -------------------------------------------------------------

@pytest.mark.parametrize("action,exposure,expected", [
    (Action.ALLOW_TRANSACTION, 0, Route.AUTO),
    (Action.MONITOR_CARD, 99999, Route.AUTO),
    (Action.VERIFY_WITH_CUSTOMER, 99999, Route.AUTO),
    (Action.CREATE_CASE, 99999, Route.AUTO),
    (Action.ESCALATE_TO_ANALYST, 99999, Route.AUTO),
    (Action.CLOSE_NO_FRAUD, 0, Route.AUTO),
    (Action.DECLINE_TRANSACTION, 0, Route.L1),
    (Action.DECLINE_TRANSACTION, 99999, Route.L1),      # exposure-independent
    (Action.BLOCK_CARD, 0, Route.L1),
    (Action.BLOCK_CARD, 2500, Route.L1),                # boundary: <= is L1
    (Action.BLOCK_CARD, 2500.01, Route.L2),
    (Action.BLOCK_ALL_CARDS, 0, Route.L2),              # always L2
    (Action.FILE_REPORT, 0, Route.L2),                  # always L2
])
def test_routes(action, exposure, expected):
    assert route_for(action, exposure) is expected


def test_only_auto_is_executable():
    assert is_executable(Action.MONITOR_CARD)
    assert not is_executable(Action.BLOCK_CARD, 100)
    assert not is_executable(Action.FILE_REPORT)


# --- section 3a: the two gates ---------------------------------------------------------------

def test_create_case_gate():
    assert should_create_case(fraud_probability=0.30, evidence_requested=False, customer_disputes=False)
    assert not should_create_case(fraud_probability=0.29, evidence_requested=False, customer_disputes=False)
    # ...but asking for evidence or a dispute opens one regardless of probability
    assert should_create_case(fraud_probability=0.01, evidence_requested=True, customer_disputes=False)
    assert should_create_case(fraud_probability=0.01, evidence_requested=False, customer_disputes=True)


def test_file_report_needs_strength_AND_an_aggravator():
    strong = dict(verdict="fraud", fraud_probability=0.9, ring_signal=False, pattern="card_not_present_fraud")
    assert not should_file_report(exposure_usd=999, **strong)     # strong but nothing aggravating
    assert should_file_report(exposure_usd=1000.01, **strong)     # exposure
    assert should_file_report(exposure_usd=10, **{**strong, "ring_signal": True})
    assert should_file_report(exposure_usd=10, **{**strong, "pattern": "undocumented"})
    # weak cases never file, however aggravated
    assert not should_file_report(verdict="uncertain", fraud_probability=0.69,
                                  exposure_usd=50000, ring_signal=True, pattern="undocumented")


# --- R1-R10 ----------------------------------------------------------------------------------

def test_r1_verify_before_blocking_on_a_single_weak_signal():
    d = evaluate(PolicyContext(fraud_probability=0.45, n_independent_signals=1, exposure_usd=300))
    assert Action.VERIFY_WITH_CUSTOMER.value in d.action_names
    assert Action.BLOCK_CARD.value not in d.action_names
    assert any(r.reason.startswith("R1") for r in d.recommendations)


def test_r2_denial_blocks_and_opens_a_case():
    d = evaluate(PolicyContext(fraud_probability=0.86, verdict="fraud", customer_response="denied",
                               exposure_usd=268.43, n_independent_signals=3))
    assert Action.BLOCK_CARD.value in d.action_names
    assert Action.CREATE_CASE.value in d.action_names
    assert Action.FILE_REPORT.value not in d.action_names      # $268 and no ring
    assert route_for(Action.BLOCK_CARD, 268.43) is Route.L1


def test_r2_denial_with_ring_also_files():
    d = evaluate(PolicyContext(fraud_probability=0.86, verdict="fraud", customer_response="denied",
                               exposure_usd=268.43, ring_signal=True, ring_element="a shared device profile",
                               n_independent_signals=3))
    assert Action.FILE_REPORT.value in d.action_names
    assert Action.MONITOR_CONNECTED_CARDS.value in d.action_names


def test_r3_confirmation_closes():
    d = evaluate(PolicyContext(fraud_probability=0.2, verdict="legitimate",
                               customer_response="confirmed", n_independent_signals=2))
    assert Action.CLOSE_NO_FRAUD.value in d.action_names
    assert Action.BLOCK_CARD.value not in d.action_names


def test_r4_no_reply_monitors_and_escalates_above_500():
    quiet = PolicyContext(customer_response="no_reply", exposure_usd=600,
                          pending_authorization=True, n_independent_signals=2)
    d = evaluate(quiet)
    assert Action.MONITOR_CARD.value in d.action_names
    assert Action.DECLINE_TRANSACTION.value in d.action_names
    assert Action.ESCALATE_TO_ANALYST.value in d.action_names
    small = evaluate(PolicyContext(customer_response="no_reply", exposure_usd=100,
                                   n_independent_signals=2))
    assert Action.ESCALATE_TO_ANALYST.value not in small.action_names


def test_r5_card_testing_escalates_once_a_large_purchase_clears():
    base = dict(card_testing=True, fraud_probability=0.8, verdict="fraud",
                exposure_usd=268, n_independent_signals=2)
    d = evaluate(PolicyContext(**base))
    assert Action.DECLINE_TRANSACTION.value in d.action_names
    assert Action.STEP_UP_AUTH.value in d.action_names
    assert Action.BLOCK_CARD.value not in d.action_names
    d2 = evaluate(PolicyContext(**base, cleared_purchase_over_100=True))
    assert Action.BLOCK_CARD.value in d2.action_names


def test_r6_shared_origin_names_the_element():
    d = evaluate(PolicyContext(fraud_probability=0.8, verdict="fraud", ring_signal=True,
                               ring_element="device profile SM-G955U | Android 8.0.0",
                               exposure_usd=400, n_independent_signals=3))
    assert Action.FILE_REPORT.value in d.action_names
    assert Action.MONITOR_CONNECTED_CARDS.value in d.action_names
    assert any("SM-G955U" in r.reason for r in d.recommendations)


def test_r7_recurring_charge_is_never_blocked():
    """The brief's explicit trap: a real dispute over the customer's own subscription."""
    d = evaluate(PolicyContext(trigger_type="customer_report", matches_recurring_pattern=True,
                               customer_response="denied", fraud_probability=0.2,
                               exposure_usd=49, n_independent_signals=2))
    assert Action.BLOCK_CARD.value not in d.action_names
    assert Action.DECLINE_TRANSACTION.value not in d.action_names
    assert Action.WARN_CUSTOMER.value in d.action_names
    assert Action.CREATE_CASE.value in d.action_names


def test_r8_uncertain_and_exposed_escalates():
    d = evaluate(PolicyContext(verdict="uncertain", exposure_usd=600, fraud_probability=0.5,
                               n_independent_signals=2))
    assert Action.ESCALATE_TO_ANALYST.value in d.action_names
    quiet = evaluate(PolicyContext(verdict="uncertain", exposure_usd=100,
                                   fraud_probability=0.5, n_independent_signals=2))
    assert Action.ESCALATE_TO_ANALYST.value not in quiet.action_names


def test_r8_conflicting_evidence_escalates_at_any_exposure():
    d = evaluate(PolicyContext(verdict="uncertain", exposure_usd=10, evidence_conflicts=True,
                               fraud_probability=0.5, n_independent_signals=2))
    assert Action.ESCALATE_TO_ANALYST.value in d.action_names


def test_r9_undocumented_files_and_escalates():
    d = evaluate(PolicyContext(pattern="undocumented", verdict="fraud", fraud_probability=0.8,
                               exposure_usd=50, n_independent_signals=3))
    assert Action.FILE_REPORT.value in d.action_names       # 3a: undocumented is an aggravator
    assert Action.ESCALATE_TO_ANALYST.value in d.action_names
    assert Action.CREATE_CASE.value in d.action_names


def test_r10_block_all_cards_requires_two_cards_or_compromised_credentials():
    one = evaluate(PolicyContext(verdict="fraud", fraud_probability=0.9, customer_response="denied",
                                 customer_n_cards=1, n_cards_confirmed_fraud=1,
                                 exposure_usd=500, n_independent_signals=3))
    assert Action.BLOCK_ALL_CARDS.value not in one.action_names

    two = evaluate(PolicyContext(verdict="fraud", fraud_probability=0.9, customer_response="denied",
                                 customer_n_cards=2, n_cards_confirmed_fraud=2,
                                 exposure_usd=500, n_independent_signals=3))
    assert Action.BLOCK_ALL_CARDS.value in two.action_names
    assert Action.BLOCK_CARD.value not in two.action_names   # subsumed


# --- composition -----------------------------------------------------------------------------

def test_actions_are_ordered_by_what_happens_first():
    d = evaluate(PolicyContext(card_testing=True, cleared_purchase_over_100=True, verdict="fraud",
                               fraud_probability=0.9, customer_response="denied",
                               exposure_usd=1500, ring_signal=True, ring_element="a shared device",
                               n_independent_signals=3))
    names = d.action_names
    assert names.index(Action.DECLINE_TRANSACTION.value) < names.index(Action.BLOCK_CARD.value)
    assert names.index(Action.BLOCK_CARD.value) < names.index(Action.CREATE_CASE.value)
    assert names.index(Action.CREATE_CASE.value) < names.index(Action.FILE_REPORT.value)


def test_no_contradictory_pairs():
    d = evaluate(PolicyContext(verdict="fraud", fraud_probability=0.9, customer_response="denied",
                               exposure_usd=200, n_independent_signals=3))
    assert not ({Action.ALLOW_TRANSACTION.value, Action.CLOSE_NO_FRAUD.value} & set(d.action_names))


def test_every_recommendation_cites_a_rule():
    """Policy section 7: every recommendation must state why it follows from the policy."""
    for ctx in [PolicyContext(fraud_probability=0.45, n_independent_signals=1),
                PolicyContext(customer_response="denied", verdict="fraud", exposure_usd=2000,
                              fraud_probability=0.9, n_independent_signals=3),
                PolicyContext(verdict="legitimate", fraud_probability=0.05, n_independent_signals=2)]:
        for rec in evaluate(ctx).recommendations:
            assert rec.reason.strip(), f"{rec.action} has no reason"


def test_a_decision_is_always_produced():
    """Even an empty context yields something defensible rather than an empty list."""
    d = evaluate(PolicyContext())
    assert d.recommendations


# --- section 6: stopping ---------------------------------------------------------------------

def test_stop_requires_two_signals_even_when_confident():
    lone = PolicyContext(fraud_probability=0.95, n_independent_signals=1)
    stop, why = should_stop(lone, loop_count=0)
    assert not stop and "single signal" in why

    pair = PolicyContext(fraud_probability=0.95, n_independent_signals=2)
    stop, _ = should_stop(pair, loop_count=0)
    assert stop


def test_stop_on_customer_response_and_on_loop_cap():
    stop, why = should_stop(PolicyContext(customer_response="denied"), loop_count=0)
    assert stop and "denied" in why
    stop, why = should_stop(PolicyContext(fraud_probability=0.5, n_independent_signals=1),
                            loop_count=3, max_loops=3)
    assert stop and "cap" in why


def test_midband_keeps_investigating():
    stop, _ = should_stop(PolicyContext(fraud_probability=0.5, n_independent_signals=2), loop_count=0)
    assert not stop
