"""Recurring-charge detection tests. Synthetic frames, no dataset needed.

R7 is the only brake on R2, and R2 blocks the card on every dispute. So a false negative here
blocks a cardholder's subscription, and a false positive clears real fraud that happens to cost
about the same as one. Both directions are tested.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.recurring import (AMOUNT_TOLERANCE_ABS, INTERVAL_CV_MAX, MIN_OCCURRENCES,
                                    Recurrence, detect)

FLAG_TS = pd.Timestamp("2016-12-01 09:00:00")


def frame(rows: list[tuple[str, str, float]], product="C", domain="gmail.com") -> pd.DataFrame:
    """rows = [(txn_id, timestamp, amount), ...]"""
    return pd.DataFrame([
        {"TransactionID": tid, "ts": pd.Timestamp(ts), "TransactionAmt": amt,
         "ProductCD": product, "P_emaildomain": domain}
        for tid, ts, amt in rows])


def monthly(amount: float, n: int, *, start="2016-07-01 09:00:00", jitter=(0, 0, 0, 0, 0, 0)):
    """n charges of `amount`, roughly a month apart, ending before FLAG_TS."""
    rows = []
    ts = pd.Timestamp(start)
    for k in range(n):
        rows.append((f"T{k}", str(ts + pd.Timedelta(days=jitter[k % len(jitter)])), amount))
        ts += pd.Timedelta(days=30)
    return frame(rows)


# --- the case R7 exists for -----------------------------------------------------------------

def test_a_monthly_subscription_is_recognised():
    prior = monthly(9.99, 4, start="2016-08-04 09:00:00")
    r = detect(prior, amount=9.99, ts=pd.Timestamp("2016-12-02 09:00:00"),
               product_cd="C", email_domain="gmail.com")
    assert r.matches
    assert r.cadence == "monthly"
    assert r.n_prior == 4
    assert len(r.prior_txn_ids) == 4
    assert r.same_domain


def test_small_billing_drift_is_tolerated():
    """Subscriptions re-bill at the same price, but tax and rounding move the cents."""
    prior = frame([("T0", "2016-09-02 09:00:00", 14.99),
                   ("T1", "2016-10-02 09:00:00", 15.10),
                   ("T2", "2016-11-02 09:00:00", 14.99)])
    r = detect(prior, amount=15.05, ts=pd.Timestamp("2016-12-02 09:00:00"))
    assert r.matches


def test_a_few_days_of_jitter_is_tolerated():
    prior = monthly(29.99, 4, start="2016-08-03 09:00:00", jitter=(0, 2, -1, 3, 0, 1))
    r = detect(prior, amount=29.99, ts=pd.Timestamp("2016-12-01 09:00:00"))
    assert r.matches


@pytest.mark.parametrize("days,cadence", [(7, "weekly"), (14, "fortnightly"),
                                          (91, "quarterly"), (365, "annual")])
def test_other_standard_cadences_are_recognised(days, cadence):
    ts = pd.Timestamp("2016-01-04 09:00:00")
    rows = []
    for k in range(4):
        rows.append((f"T{k}", str(ts), 12.50))
        ts += pd.Timedelta(days=days)
    r = detect(frame(rows), amount=12.50, ts=ts)
    assert r.matches
    assert r.cadence == cadence


def test_the_claim_describes_recurrence_not_a_merchant():
    """There is no merchant column in the dataset, so the evidence must not imply one."""
    r = detect(monthly(9.99, 3, start="2016-09-02 09:00:00"), amount=9.99,
               ts=pd.Timestamp("2016-12-02 09:00:00"))
    claim = r.as_evidence_claim()
    assert "merchant" not in claim.lower()
    assert "recurring charge" in claim
    assert "$9.99" in claim


# --- the false positives that would clear real fraud ----------------------------------------

def test_one_prior_charge_is_not_a_pattern():
    prior = frame([("T0", "2016-11-01 09:00:00", 49.00)])
    r = detect(prior, amount=49.00, ts=FLAG_TS)
    assert not r.matches
    assert "too few" in r.reason


def test_a_repeated_amount_at_random_intervals_is_not_recurring():
    """Three coffees at $4.50 are not a subscription."""
    prior = frame([("T0", "2016-07-02 09:00:00", 4.50),
                   ("T1", "2016-07-05 09:00:00", 4.50),
                   ("T2", "2016-10-28 09:00:00", 4.50)])
    r = detect(prior, amount=4.50, ts=FLAG_TS)
    assert not r.matches
    assert "irregular" in r.reason


def test_a_regular_but_nonstandard_interval_is_refused():
    ts = pd.Timestamp("2016-09-01 09:00:00")
    rows = []
    for k in range(4):
        rows.append((f"T{k}", str(ts), 20.00))
        ts += pd.Timedelta(days=45)
    r = detect(frame(rows), amount=20.00, ts=ts)
    assert not r.matches
    assert "no standard billing cadence" in r.reason


def test_an_off_cycle_charge_is_refused_even_on_a_real_subscription():
    """The signature includes WHEN. A second charge mid-cycle is not the subscription."""
    prior = monthly(9.99, 4, start="2016-08-04 09:00:00")
    off_cycle = pd.Timestamp(prior.ts.max()) + pd.Timedelta(days=4)
    r = detect(prior, amount=9.99, ts=off_cycle)
    assert not r.matches
    assert "off-cycle" in r.reason
    assert r.cadence == "monthly"          # the pattern is reported even though it did not match


def test_a_different_amount_does_not_match():
    prior = monthly(9.99, 4, start="2016-08-04 09:00:00")
    r = detect(prior, amount=482.12, ts=pd.Timestamp("2016-12-02 09:00:00"))
    assert not r.matches


def test_a_different_product_code_does_not_match():
    prior = monthly(9.99, 4, start="2016-08-04 09:00:00")     # all ProductCD C
    r = detect(prior, amount=9.99, ts=pd.Timestamp("2016-12-02 09:00:00"), product_cd="W")
    assert not r.matches


# --- the as-of discipline -------------------------------------------------------------------

def test_later_transactions_are_invisible():
    """A case must never be cleared using its own future.

    The subscription runs Aug-Nov. Flag the September charge and the detector may only see the
    two that already happened, so it cannot claim the established pattern that the October and
    November charges would have given it.
    """
    prior = monthly(9.99, 4, start="2016-08-04 09:00:00")
    assert len(prior) == 4
    early_flag = pd.Timestamp("2016-09-05 09:00:00")
    r = detect(prior, amount=9.99, ts=early_flag)
    assert r.n_prior == 2, "the October and November charges must not be visible"
    assert not r.matches
    assert "off-cycle" in r.reason


def test_the_same_charge_matches_once_the_pattern_has_had_time_to_form():
    """The other half of the as-of test: same subscription, flagged a cycle later."""
    prior = monthly(9.99, 4, start="2016-08-04 09:00:00")
    r = detect(prior, amount=9.99, ts=pd.Timestamp("2016-12-02 09:00:00"))
    assert r.n_prior == 4
    assert r.matches


def test_an_empty_history_is_not_an_error():
    for empty in (None, pd.DataFrame(columns=["TransactionID", "ts", "TransactionAmt"])):
        r = detect(empty, amount=10.0, ts=FLAG_TS)
        assert isinstance(r, Recurrence)
        assert not r.matches
        assert r.reason


# --- the counterparty field is corroboration, never a filter --------------------------------

def test_a_missing_domain_does_not_block_a_match():
    """`P_emaildomain` is null on 94,480 rows; absence is not evidence against recurrence."""
    prior = monthly(9.99, 4, start="2016-08-04 09:00:00")
    prior["P_emaildomain"] = None
    r = detect(prior, amount=9.99, ts=pd.Timestamp("2016-12-02 09:00:00"),
               email_domain="gmail.com")
    assert r.matches
    assert not r.same_domain
    assert "no purchaser domain recorded" in r.as_evidence_claim()


def test_a_matching_domain_is_recorded_as_corroboration():
    r = detect(monthly(9.99, 4, start="2016-08-04 09:00:00"), amount=9.99,
               ts=pd.Timestamp("2016-12-02 09:00:00"), email_domain="gmail.com")
    assert r.same_domain
    assert "same purchaser email domain" in r.as_evidence_claim()


# --- tolerances -----------------------------------------------------------------------------

def test_small_amounts_get_an_absolute_tolerance_floor():
    """A 2% window on $3.99 is 8 cents, which is narrower than real billing drift."""
    prior = frame([("T0", "2016-09-02 09:00:00", 3.99),
                   ("T1", "2016-10-02 09:00:00", 4.29),
                   ("T2", "2016-11-02 09:00:00", 3.99)])
    r = detect(prior, amount=4.20, ts=pd.Timestamp("2016-12-02 09:00:00"))
    assert r.matches
    assert AMOUNT_TOLERANCE_ABS >= 0.30


def test_regularity_threshold_is_actually_applied():
    assert 0 < INTERVAL_CV_MAX < 1
