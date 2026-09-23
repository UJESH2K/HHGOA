"""Rubric tests. No dataset, no credentials - these run anywhere.

The tests worth reading are the ones that pin the rubric's *refusals*: that `id_15 = New` can
never be the signal that licenses a verdict, that `risk_score` can never be either, and that a
decisive probability on one family of evidence is held back rather than acted on. Those are the
three findings that shaped the whole scoring design, and a future weight change that breaks one
of them should fail a test rather than quietly change twenty answer files.
"""
from __future__ import annotations

import math

import pytest

from src.scoring.rubric import (AMOUNT, AMT_ANCHORS, BAND_LOW, BASE_PRIOR, CONFIRMED_CAP,
                                FIT, FRAUD_AT, LEGITIMATE_AT, MATERIAL, RubricInput, _interp,
                                score)


# --- the neutral case -----------------------------------------------------------------------

def test_no_observations_returns_the_prior():
    s = score(RubricInput())
    assert s.probability == pytest.approx(BASE_PRIOR, abs=1e-9)
    assert s.n_independent_signals == 0
    assert s.verdict == "uncertain"
    assert not s.clamped


def test_scoring_is_deterministic():
    i = RubricInput(amt_pctile_prior=0.97, n_prior_txns=100, amount_usd=482.12,
                    trigger_type="customer_report")
    assert score(i).to_json() == score(i).to_json()


# --- independence: the rule that protects against overconfidence ---------------------------

def test_one_strong_signal_cannot_produce_a_verdict():
    """HHG-002 / HHG-010 shape: top-percentile amount, nothing else.

    The strongest single signal in the rubric (+1.60 log-odds) lifts a 0.30 prior to ~0.68,
    which is deliberately just short of the fraud threshold: no one family can get there alone.
    """
    s = score(RubricInput(amt_pctile_prior=1.0, n_prior_txns=35, amount_usd=292.36))
    assert s.n_independent_signals == 1
    assert s.probability > BASE_PRIOR + 0.30       # the evidence IS strong
    assert s.probability < FRAUD_AT                # but not strong enough on its own
    assert s.verdict == "uncertain"


def test_a_decisive_single_family_is_flagged_as_blocked():
    """Card testing can clear 0.70 on its own, so the independence guard has to say so."""
    s = score(RubricInput(card_testing=True, card_testing_escalated=True))
    assert s.n_independent_signals == 1
    assert s.probability >= FRAUD_AT
    assert s.verdict == "uncertain"
    assert s.verdict_blocked_by_independence


def test_two_families_unlock_a_verdict():
    """HHG-015 shape: top-percentile amount AND a region the card has never used."""
    s = score(RubricInput(amt_pctile_prior=1.0, n_prior_txns=68, amount_usd=599.94,
                          new_region=True))
    assert s.n_independent_signals == 2
    assert s.verdict == "fraud"
    assert not s.verdict_blocked_by_independence


def test_probability_is_clamped_into_the_band_on_a_single_signal():
    s = score(RubricInput(amt_pctile_prior=0.02, n_prior_txns=900, amount_usd=30.91))
    assert s.clamped
    assert s.probability == pytest.approx(BAND_LOW)


def test_clamp_does_not_fire_once_two_families_agree():
    s = score(RubricInput(amt_pctile_prior=0.02, n_prior_txns=900, amount_usd=30.91,
                          known_region=True, known_product=True, known_email_domain=True,
                          timing_typical=True, device_state="Found"))
    assert not s.clamped
    assert s.probability < BAND_LOW
    assert s.verdict == "legitimate"


# --- the two signals that must never be evidence on their own ------------------------------

def test_device_state_new_is_never_material():
    """43% of all identity records are `New`, and 11 of the 20 exam cases carry it."""
    s = score(RubricInput(device_state="New"))
    (c,) = [c for c in s.contributions if c.signal == "device_state_new"]
    assert abs(c.log_odds) < MATERIAL
    assert s.n_independent_signals == 0
    assert s.verdict == "uncertain"


@pytest.mark.parametrize("rs", [0.0, 0.05, 0.25, 0.5, 0.7, 0.76, 0.9, 0.99, 1.0])
def test_risk_score_can_never_be_an_independent_signal(rs):
    """The brief calls the score an input, not a verdict. Enforced by weight, not by prompt."""
    s = score(RubricInput(risk_score=rs))
    model = [c for c in s.contributions if c.signal == "risk_score"]
    assert all(abs(c.log_odds) < MATERIAL for c in model)
    assert s.n_independent_signals == 0


def test_a_high_risk_score_alone_cannot_reach_the_fraud_threshold():
    s = score(RubricInput(risk_score=0.99, device_state="New"))
    assert s.probability < FRAUD_AT
    assert s.verdict == "uncertain"


def test_prior_fraud_history_corroborates_but_does_not_decide():
    """16 of 20 exam customers have history, and that history is 83.8% fraud."""
    s = score(RubricInput(prior_confirmed_fraud=19))
    (c,) = [c for c in s.contributions if c.signal == "prior_confirmed_fraud"]
    assert 0 < c.log_odds < MATERIAL
    assert s.n_independent_signals == 0


# --- amount behaviour -----------------------------------------------------------------------

@pytest.mark.parametrize("lo,hi", list(zip([0.0, 0.1, 0.3, 0.5, 0.85, 0.93, 0.97],
                                           [0.1, 0.3, 0.5, 0.85, 0.93, 0.97, 1.0])))
def test_amount_weight_is_monotonic_in_percentile(lo, hi):
    assert _interp(lo, AMT_ANCHORS) <= _interp(hi, AMT_ANCHORS)


def test_amount_weight_has_no_cliffs():
    """The bug this replaced: 0.25 scored -1.20 and 0.28 scored 0.00."""
    step = max(abs(_interp(x / 1000, AMT_ANCHORS) - _interp((x + 1) / 1000, AMT_ANCHORS))
               for x in range(1000))
    assert step < 0.05


def test_amount_percentile_is_withheld_on_a_thin_history():
    """A percentile over four transactions is noise dressed as a statistic."""
    s = score(RubricInput(amt_pctile_prior=1.0, n_prior_txns=3, amount_usd=1000.0))
    (c,) = [c for c in s.contributions if c.family == AMOUNT]
    assert c.log_odds == 0.0
    assert "too few" in c.note


def test_missing_percentile_is_recorded_not_silently_skipped():
    s = score(RubricInput(amt_pctile_prior=None))
    assert [c for c in s.contributions if c.family == AMOUNT]


# --- profile fit: the exonerating family ---------------------------------------------------

def _consistent(**kw) -> RubricInput:
    base = dict(n_prior_txns=200, amt_pctile_prior=0.5, amount_usd=60.0,
                known_region=True, known_product=True, known_email_domain=True,
                timing_typical=True, device_state="Found")
    base.update(kw)
    return RubricInput(**base)


def test_consistency_across_several_dimensions_lowers_the_probability():
    s = score(_consistent())
    (c,) = [c for c in s.contributions if c.family == FIT]
    assert c.log_odds < -MATERIAL
    assert s.probability < BASE_PRIOR


def test_profile_fit_refuses_to_fire_when_anything_is_novel():
    """Consistency is exonerating in the ABSENCE of an anomaly, never in spite of one."""
    for novelty in (dict(new_region=True), dict(new_product=True),
                    dict(new_email_domain=True), dict(new_time_of_day=True),
                    dict(ring_signal=True, ring_n_cards=3), dict(card_testing=True),
                    dict(proxy="ANONYMOUS"), dict(step_up_result="failed")):
        s = score(_consistent(**novelty))
        assert not [c for c in s.contributions if c.family == FIT], novelty


def test_profile_fit_needs_real_history():
    assert not [c for c in score(_consistent(n_prior_txns=19)).contributions
                if c.family == FIT]
    assert [c for c in score(_consistent(n_prior_txns=20)).contributions if c.family == FIT]


def test_a_burst_blocks_the_pace_check():
    s = score(_consistent(prior_txns_1h=4))
    fit = [c for c in s.contributions if c.family == FIT]
    assert fit and "normal pace" not in fit[0].note


# --- customer response ----------------------------------------------------------------------

def test_customer_report_trigger_counts_as_a_dispute_without_asking_again():
    """R2 applies on the trigger alone - the customer has already said it is not theirs."""
    a = score(RubricInput(trigger_type="customer_report"))
    b = score(RubricInput(trigger_type="risk_score", customer_response="denied"))
    assert [c.log_odds for c in a.contributions] == [c.log_odds for c in b.contributions]


def test_a_confirmation_settles_the_question():
    """R3 is unconditional, so anomaly signals must not be able to out-vote it."""
    s = score(RubricInput(amt_pctile_prior=0.99, n_prior_txns=100, amount_usd=500.0,
                          new_region=True, customer_response="confirmed"))
    assert s.probability <= CONFIRMED_CAP
    assert s.verdict == "legitimate"
    assert s.override.startswith("R3")


@pytest.mark.parametrize("ato", [dict(ring_signal=True, ring_n_cards=3),
                                 dict(card_testing=True),
                                 dict(step_up_result="failed")])
def test_a_confirmation_does_not_clear_a_compromised_account(ato):
    """If the account looks taken over, whoever answered may not be the cardholder."""
    s = score(RubricInput(amt_pctile_prior=0.99, n_prior_txns=100, amount_usd=500.0,
                          customer_response="confirmed", **ato))
    assert not s.override
    assert s.probability > CONFIRMED_CAP


def test_recurring_pattern_brakes_a_dispute():
    """R7: a disputed charge matching the cardholder's own recurring pattern is not fraud."""
    disputed = score(RubricInput(trigger_type="customer_report", amt_pctile_prior=0.5,
                                 n_prior_txns=100))
    braked = score(RubricInput(trigger_type="customer_report", amt_pctile_prior=0.5,
                               n_prior_txns=100, matches_recurring_pattern=True))
    assert braked.probability < disputed.probability
    assert braked.probability < BASE_PRIOR


def test_step_up_outcomes_move_in_opposite_directions():
    passed = score(RubricInput(step_up_result="passed"))
    failed = score(RubricInput(step_up_result="failed"))
    assert passed.probability < BASE_PRIOR < failed.probability


# --- shared origin and card testing ---------------------------------------------------------

def test_a_rare_shared_device_is_strong_evidence():
    s = score(RubricInput(ring_signal=True, ring_n_cards=4))
    (c,) = [c for c in s.contributions if c.signal == "shared_device_profile"]
    assert c.log_odds > MATERIAL


def test_a_volume_artefact_ring_is_recorded_but_discounted():
    """HHG-018 sits in nine "rings" purely because it has 7,091 transactions."""
    s = score(RubricInput(ring_signal=True, ring_volume_artefact=True, ring_n_cards=9))
    (c,) = [c for c in s.contributions if c.signal == "shared_device_profile"]
    assert 0 < c.log_odds < MATERIAL
    assert "artefact" in c.note


def test_card_testing_is_strong_and_independent():
    s = score(RubricInput(card_testing=True, card_testing_escalated=True))
    assert s.n_independent_signals >= 1
    assert s.probability > 0.70


# --- output shape ---------------------------------------------------------------------------

def test_drivers_are_the_top_three_material_contributions_by_magnitude():
    s = score(RubricInput(amt_pctile_prior=1.0, n_prior_txns=500, amount_usd=900.0,
                          new_region=True, new_product=True, proxy="ANONYMOUS",
                          trigger_type="customer_report", prior_confirmed_fraud=3))
    assert len(s.drivers) == 3
    mags = [abs(c.log_odds) for c in s.drivers]
    assert mags == sorted(mags, reverse=True)
    assert all(c.material for c in s.drivers)


def test_every_contribution_can_be_phrased_as_an_evidence_claim():
    s = score(RubricInput(amt_pctile_prior=0.99, n_prior_txns=100, amount_usd=482.12,
                          new_region=True))
    for c in s.contributions:
        claim = c.as_evidence_claim()
        assert claim and c.family in claim
        assert ("raises" in claim) or ("lowers" in claim)


def test_to_json_carries_what_the_answer_file_needs():
    s = score(RubricInput(amt_pctile_prior=0.97, n_prior_txns=101, amount_usd=482.12,
                          trigger_type="customer_report"))
    j = s.to_json()
    assert set(j) == {"fraud_probability", "prior", "n_independent_signals", "families",
                      "clamped_to_band", "override", "drivers", "all_contributions"}
    assert 0.0 <= j["fraud_probability"] <= 1.0
    assert j["drivers"] and all(set(d) == {"signal", "family", "log_odds", "note"}
                                for d in j["drivers"])


def test_explain_shows_the_prior_and_every_contribution():
    s = score(RubricInput(amt_pctile_prior=0.97, n_prior_txns=101, amount_usd=482.12,
                          new_region=True, device_state="New", risk_score=0.25))
    text = s.explain()
    assert "prior" in text
    for c in s.contributions:
        assert c.signal in text
    assert f"{s.probability:.2f}" in text


@pytest.mark.parametrize("field,value", [
    ("amt_pctile_prior", 0.0), ("amt_pctile_prior", 1.0), ("risk_score", 0.0),
    ("risk_score", 1.0), ("prior_txns_1h", 10_000), ("prior_confirmed_fraud", 356),
])
def test_extremes_stay_in_range(field, value):
    s = score(RubricInput(**{field: value}, n_prior_txns=100, amount_usd=100.0))
    assert 0.0 < s.probability < 1.0
    assert math.isfinite(s.probability)
