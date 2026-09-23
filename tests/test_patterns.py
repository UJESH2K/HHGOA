"""Pattern classification tests. No dataset, no credentials.

The two that matter most are structural: a card-present transaction can never be labelled
card-not-present, and `undocumented` cannot be reached without coordinated activity across
accounts. The first protects six of the 20 exam cases from a definitionally impossible label;
the second stops the agent filing regulatory reports whenever it feels uncertain.
"""
from __future__ import annotations

import pytest

from src.scoring.patterns import (ALL_PATTERNS, ATO, CARD_PRESENT_IMPOSSIBLE, CARD_TESTING, CNP,
                                  CNP_NEW_DEVICE, NONE, OUT_OF_REGION, UNDOCUMENTED, classify)
from src.scoring.rubric import RubricInput, score


def verdict_for(**kw):
    """Classify an input, defaulting to a shape that is suspicious enough to carry a pattern."""
    base = dict(amt_pctile_prior=0.99, n_prior_txns=200, amount_usd=500.0,
                trigger_type="customer_report")
    base.update(kw)
    i = RubricInput(**base)
    return i, classify(i, score(i))


# --- structural impossibility ---------------------------------------------------------------

@pytest.mark.parametrize("pattern", CARD_PRESENT_IMPOSSIBLE)
def test_an_in_person_transaction_is_never_card_not_present(pattern):
    """`channel` is a restatement of ProductCD: all 439,670 `W` rows are in person."""
    _, v = verdict_for(channel="in_person", product_cd="W")
    assert v.pattern != pattern
    rejected = next(c for c in v.candidates if c.pattern == pattern)
    assert rejected.rejected
    assert "impossible by definition" in rejected.rejected


def test_an_in_person_case_with_a_new_region_is_out_of_region():
    _, v = verdict_for(channel="in_person", product_cd="W", new_region=True)
    assert v.pattern == OUT_OF_REGION


def test_a_missing_region_is_not_a_new_one():
    """94.8% of C-product transactions carry no billing region at all."""
    _, v = verdict_for(new_region=False, known_region=False)
    rejected = next(c for c in v.candidates if c.pattern == OUT_OF_REGION)
    assert rejected.rejected
    assert "not a new one" in rejected.rationale


# --- the documented online patterns ---------------------------------------------------------

def test_a_known_device_online_case_is_plain_card_not_present():
    _, v = verdict_for(device_state="Found")
    assert v.pattern == CNP


def test_a_new_device_online_case_is_the_new_device_variant():
    _, v = verdict_for(device_state="New")
    assert v.pattern == CNP_NEW_DEVICE


def test_a_new_device_with_session_evidence_is_account_takeover():
    """What separates takeover from card misuse is evidence of control of the session."""
    _, v = verdict_for(device_state="New", proxy="ANONYMOUS")
    assert v.pattern == ATO
    assert "control of the account" in v.rationale


@pytest.mark.parametrize("extra", [dict(prior_txns_1h=4), dict(n_affected_txns=3),
                                   dict(step_up_result="failed")])
def test_takeover_can_be_supported_by_any_session_signal(extra):
    _, v = verdict_for(device_state="New", **extra)
    assert v.pattern == ATO


def test_a_new_device_alone_is_not_takeover():
    """`id_15 = New` covers 43% of all device records and 11 of the 20 exam cases."""
    _, v = verdict_for(device_state="New")
    rejected = next(c for c in v.candidates if c.pattern == ATO)
    assert rejected.rejected
    assert "43%" in rejected.rejected


def test_session_evidence_without_a_new_device_ranks_takeover_lower_than_cnp():
    _, v = verdict_for(device_state="Found", proxy="HIDDEN")
    ato = next(c for c in v.candidates if c.pattern == ATO)
    cnp = next(c for c in v.candidates if c.pattern == CNP)
    assert ato.viable and ato.score < cnp.score
    assert v.pattern == CNP


# --- card testing ---------------------------------------------------------------------------

def test_the_detector_owns_the_card_testing_label():
    _, fired = verdict_for(card_testing=True)
    assert fired.pattern == CARD_TESTING

    _, not_fired = verdict_for(card_testing=False)
    rejected = next(c for c in not_fired.candidates if c.pattern == CARD_TESTING)
    assert rejected.rejected
    assert "28 cards" in rejected.rejected


def test_card_testing_outranks_every_other_pattern_when_it_fires():
    _, v = verdict_for(card_testing=True, device_state="New", proxy="ANONYMOUS",
                       new_region=True)
    assert v.pattern == CARD_TESTING


# --- undocumented: deliberately hard to reach -----------------------------------------------

def test_undocumented_needs_a_real_shared_origin():
    _, v = verdict_for(device_state="New", proxy="ANONYMOUS")
    assert v.pattern != UNDOCUMENTED


def test_undocumented_is_refused_for_a_volume_artefact_ring():
    """HHG-018 lands in nine "rings" purely because it has 7,091 transactions."""
    _, v = verdict_for(ring_signal=True, ring_volume_artefact=True, ring_n_cards=9)
    assert v.pattern != UNDOCUMENTED
    rejected = next(c for c in v.candidates if c.pattern == UNDOCUMENTED)
    assert "volume artefact" in rejected.rejected


def test_undocumented_is_refused_when_the_assessment_is_not_confident():
    i = RubricInput(amt_pctile_prior=0.5, n_prior_txns=200, amount_usd=60.0,
                    ring_signal=True, ring_n_cards=3)
    v = classify(i, score(i))
    assert v.pattern != UNDOCUMENTED


def test_a_qualified_ring_on_a_confident_case_is_undocumented_and_described():
    _, v = verdict_for(ring_signal=True, ring_n_cards=4, new_region=True,
                       device_state="New")
    assert v.pattern == UNDOCUMENTED
    assert v.description
    assert "none of the five documented typologies" in v.description


def test_only_undocumented_carries_a_description():
    """The schema wants `pattern_description` empty on a documented pattern."""
    for kw in (dict(device_state="New"), dict(device_state="Found"), dict(new_region=True),
               dict(card_testing=True), dict(channel="in_person")):
        _, v = verdict_for(**kw)
        if v.pattern != UNDOCUMENTED:
            assert v.description == "", v.pattern


# --- legitimate cases claim no pattern ------------------------------------------------------

def test_a_legitimate_assessment_claims_no_pattern():
    """The schema requires that a legitimate verdict carries no episode."""
    i = RubricInput(amt_pctile_prior=0.02, n_prior_txns=900, amount_usd=30.91,
                    known_region=True, known_product=True, known_email_domain=True,
                    timing_typical=True, device_state="Found")
    s = score(i)
    assert s.verdict == "legitimate"
    assert classify(i, s).pattern == NONE


def test_a_confirmed_transaction_claims_no_pattern_even_with_anomalies():
    i = RubricInput(amt_pctile_prior=1.0, n_prior_txns=100, amount_usd=900.0,
                    new_region=True, device_state="New", customer_response="confirmed")
    assert classify(i, score(i)).pattern == NONE


# --- output shape ---------------------------------------------------------------------------

def test_the_pattern_is_always_in_the_schema_vocabulary():
    for kw in (dict(), dict(channel="in_person"), dict(card_testing=True),
               dict(ring_signal=True, ring_n_cards=3), dict(device_state="New"),
               dict(customer_response="confirmed"), dict(new_region=True)):
        _, v = verdict_for(**kw)
        assert v.pattern in ALL_PATTERNS


def test_every_candidate_explains_itself():
    _, v = verdict_for(device_state="New")
    for c in v.candidates:
        assert c.rationale or c.rejected


def test_the_runner_up_is_named_so_the_choice_can_be_defended():
    _, v = verdict_for(device_state="New")
    assert v.runner_up is not None
    assert v.runner_up.pattern != v.pattern
    assert "closest alternative" in v.why_not_runner_up()


def test_to_json_lists_the_alternatives_without_repeating_the_choice():
    _, v = verdict_for(device_state="New", proxy="ANONYMOUS")
    j = v.to_json()
    assert set(j) == {"pattern", "pattern_description", "rationale", "alternatives"}
    assert all(a["pattern"] != j["pattern"] for a in j["alternatives"])
    assert [a["score"] for a in j["alternatives"]] == sorted(
        (a["score"] for a in j["alternatives"]), reverse=True)


def test_classification_is_deterministic():
    i, v = verdict_for(device_state="New", proxy="HIDDEN")
    assert classify(i, score(i)).to_json() == v.to_json()
