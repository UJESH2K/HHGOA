"""Validator tests: start from a known-good answer for a real case, then break it one way at a
time and assert the corresponding invariant fires.

Requires `python -m src.features.build` to have run (needs data/*.parquet).
"""
from __future__ import annotations

import copy
import os

import pytest

from src.io.answer import (SAR, Answer, Case, Evidence, EvidenceRequest,
                           activity_dates_from, exposure_from)
from src.io.validate import DATA, Dataset, validate_answer, validate_set
from src.policy.actions import Action, Recommendation, Route, Decision

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "txns.parquet")),
    reason="run `python -m src.features.build` first")

FLAGGED = "3476682"        # HHG-006, C07297-K1, $482.12, online, risk 0.25


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return Dataset()


@pytest.fixture
def good(ds) -> dict:
    """A realistic, fully valid answer for HHG-006: fraud, one transaction, no report."""
    affected = [FLAGGED]
    exposure = exposure_from(affected, ds.amounts)
    ans = Answer(
        case_id="HHG-006",
        case=Case(
            status="closed_fraud", verdict="fraud", fraud_probability=0.82,
            pattern="card_not_present_new_device", pattern_description="",
            affected_txn_ids=affected, first_suspicious_txn_id=FLAGGED,
            connected_card_ids=[], connected_device_profiles=[],
            exposure_usd=exposure,
            evidence=[
                Evidence(claim="$482.12 sits at the 97th percentile of this card's own history",
                         source="graph", ref="query:card_baseline(card_id=C07297-K1)",
                         entity_ids=[FLAGGED]),
                Evidence(claim="Cardholder states they did not make the purchase",
                         source="customer", ref="evidence_request:1", entity_ids=[]),
            ],
            similar_prior_cases=[], summary="Online purchase far above this cardholder's "
            "normal range, from a device marked New, which the customer denies. Card "
            "compromised; blocked and reissued.",
            written_to_graph=True, graph_case_id="CASE-2016-0006"),
        evidence_requests=[EvidenceRequest("customer_validation", 3,
                                           "Customer states they did not make this purchase")],
        initial_actions=Decision([
            Recommendation(Action.VERIFY_WITH_CUSTOMER, Route.AUTO,
                           "R1: single signal at probability 0.55, verify before blocking")]),
        final_actions=Decision([
            Recommendation(Action.BLOCK_CARD, Route.L1, "R2: customer denied; exposure under $2,500"),
            Recommendation(Action.CREATE_CASE, Route.AUTO, "R2")]),
        what_changed="Customer denial raised probability from 0.55 to 0.82 and confirmed the block.",
        sar=SAR.not_filed("Policy 3a: exposure $482.12 is under $1,000 and the case connects to "
                          "no shared device, region or other customer"),
        stop_reason="Customer denial settled the verdict.",
        tool_calls=7, tokens=9100, latency_s=14.2)
    return ans.to_json()


def errors(findings):
    return [f for f in findings if f.level == "ERROR"]


def rules_fired(findings):
    return {f.rule for f in findings if f.level == "ERROR"}


def test_the_good_answer_is_clean(good, ds):
    assert errors(validate_answer(good, ds)) == []


def test_missing_field_is_caught(good, ds):
    for field in ("stop_reason", "tool_calls", "sar"):
        broken = copy.deepcopy(good)
        del broken[field]
        assert "I1-structure" in rules_fired(validate_answer(broken, ds)), field


def test_sar_disagreement(good, ds):
    broken = copy.deepcopy(good)
    broken["sar"]["file"] = True                      # but FILE_REPORT is not in final actions
    broken["sar"]["narrative"] = "x. " * 8
    broken["sar"]["subjects"] = ["C07297"]
    broken["sar"]["total_amount_usd"] = broken["case"]["exposure_usd"]
    broken["sar"]["activity_dates"] = ["2016-11-22", "2016-11-22"]
    assert "I2-sar-agreement" in rules_fired(validate_answer(broken, ds))


def test_legitimate_verdict_must_be_empty(good, ds):
    broken = copy.deepcopy(good)
    broken["case"]["verdict"] = "legitimate"          # while keeping txns and exposure
    assert "I3-legitimate" in rules_fired(validate_answer(broken, ds))


def test_exposure_must_match_the_cited_transactions(good, ds):
    broken = copy.deepcopy(good)
    broken["case"]["exposure_usd"] = 999.99
    assert "I4-exposure" in rules_fired(validate_answer(broken, ds))


def test_fabricated_ids_are_caught(good, ds):
    for path, value in (("affected_txn_ids", ["9999999999"]),
                        ("connected_card_ids", ["C99999-K7"]),
                        ("similar_prior_cases", ["CC-9999"])):
        broken = copy.deepcopy(good)
        broken["case"][path] = value
        if path == "affected_txn_ids":
            broken["case"]["exposure_usd"] = 0.0
            broken["case"]["first_suspicious_txn_id"] = ""
        assert "I5-ids" in rules_fired(validate_answer(broken, ds)), path


def test_undocumented_pattern_needs_a_description(good, ds):
    broken = copy.deepcopy(good)
    broken["case"]["pattern"] = "undocumented"
    assert "I6-undocumented" in rules_fired(validate_answer(broken, ds))


def test_wrong_route_is_caught(good, ds):
    broken = copy.deepcopy(good)
    broken["next_best_actions"]["final"][0]["route"] = "auto"   # BLOCK_CARD is never auto
    assert "I7-routes" in rules_fired(validate_answer(broken, ds))


def test_block_card_flips_to_L2_above_2500(good, ds):
    """Same action, different route, purely because exposure crossed the threshold."""
    broken = copy.deepcopy(good)
    broken["case"]["affected_txn_ids"] = []            # exposure recomputation is skipped
    broken["case"]["exposure_usd"] = 3000.0
    broken["case"]["first_suspicious_txn_id"] = ""
    assert "I7-routes" in rules_fired(validate_answer(broken, ds))
    broken["next_best_actions"]["final"][0]["route"] = "L2"
    assert "I7-routes" not in rules_fired(validate_answer(broken, ds))


def test_reason_is_mandatory_on_every_action(good, ds):
    broken = copy.deepcopy(good)
    broken["next_best_actions"]["final"][0]["reason"] = ""
    assert "I7-routes" in rules_fired(validate_answer(broken, ds))


def test_no_request_means_nothing_changed(good, ds):
    broken = copy.deepcopy(good)
    broken["evidence_requests"] = []                   # but initial != final still
    fired = rules_fired(validate_answer(broken, ds))
    assert "I8-evidence" in fired


def test_sar_false_must_be_empty(good, ds):
    broken = copy.deepcopy(good)
    broken["sar"]["narrative"] = "Something suspicious happened."
    assert "I9-sar-empty" in rules_fired(validate_answer(broken, ds))


def test_filed_sar_must_be_internally_consistent(good, ds):
    filed = copy.deepcopy(good)
    filed["next_best_actions"]["final"].append(
        {"action": "FILE_REPORT", "route": "L2", "reason": "R6: shared device profile"})
    filed["sar"] = {"file": True, "reason": "R6", "narrative": "s. " * 10,
                    "subjects": ["C07297", "C07297-K1"],
                    "total_amount_usd": 1.0,                       # wrong
                    "activity_dates": ["2016-01-01", "2016-01-02"]}  # wrong
    fired = rules_fired(validate_answer(filed, ds))
    assert "I10-sar-content" in fired

    filed["sar"]["total_amount_usd"] = filed["case"]["exposure_usd"]
    d = ds.timestamps[FLAGGED][:10]
    filed["sar"]["activity_dates"] = [d, d]
    assert "I10-sar-content" not in rules_fired(validate_answer(filed, ds))


def test_block_all_cards_needs_two_cards(good, ds):
    broken = copy.deepcopy(good)
    broken["next_best_actions"]["final"].append(
        {"action": "BLOCK_ALL_CARDS", "route": "L2", "reason": "R10"})
    fired = rules_fired(validate_answer(broken, ds))
    n_cards = ds.cards_per_customer.get("C07297", 0)
    assert ("I11-r10" in fired) == (n_cards < 2)


# --- set-level ------------------------------------------------------------------------------

def test_set_flags_missing_cases(good, ds):
    findings = validate_set({"HHG-006": good}, ds)
    assert sum(1 for f in findings if f.rule == "I12-set" and f.level == "ERROR") >= 19


def test_set_flags_all_fraud(good, ds):
    answers = {}
    for i in range(1, 21):
        a = copy.deepcopy(good)
        a["case_id"] = f"HHG-{i:03d}"
        answers[a["case_id"]] = a
    findings = validate_set(answers, ds)
    assert any("every verdict is the same" in f.message for f in findings)


def test_set_flags_over_filing(good, ds):
    answers = {}
    for i in range(1, 21):
        a = copy.deepcopy(good)
        a["case_id"] = f"HHG-{i:03d}"
        a["case"]["verdict"] = "legitimate" if i % 2 else "fraud"
        if i % 2 == 0:
            a["sar"]["file"] = True
        answers[a["case_id"]] = a
    findings = validate_set(answers, ds)
    assert any("file a report" in f.message for f in findings)
