"""LocalBackend tests, with the as-of discipline as the centrepiece.

A backend that leaks post-`as_of` data inflates the closed-case backtest and teaches us nothing,
so the leak tests below matter more than the lookup tests.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.graph.local import DATA, LocalBackend

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "txns.parquet")),
    reason="run `python -m src.features.build` first")

FLAGGED = "3476682"          # HHG-006  C07297  $482.12  online  2016-11-22
BIG_CUSTOMER = "C11923"      # HHG-011, 10,361 transactions


@pytest.fixture(scope="module")
def be() -> LocalBackend:
    return LocalBackend()


def test_transaction_lookup(be):
    t = be.get_transaction(FLAGGED)
    assert t["customer_id"] == "C07297"
    assert round(t["TransactionAmt"], 2) == 482.12
    assert t["channel"] == "online"


def test_resolve_card_three_ways(be):
    by_txn = be.resolve_card(txn_id=FLAGGED)
    assert by_txn.customer_id == "C07297"
    by_id = be.resolve_card(card_id=by_txn.card_id)
    by_key = be.resolve_card(card_key=by_txn.card_key)
    assert by_txn.card_key == by_id.card_key == by_key.card_key


def test_unknown_cards_are_not_citable(be):
    """The K-suffix is not derivable; cards we never saw named must not reach an answer file."""
    unknown = [c for c in be.cards.itertuples() if c.provenance == "unknown"]
    assert unknown, "expected some unresolved cards"
    s = be.resolve_card(card_key=unknown[0].card_key)
    assert not s.citable


def test_single_card_customers_are_citable(be):
    s = be.resolve_card(txn_id=FLAGGED)
    assert s.citable and s.card_id.endswith(("-K1", "-K2", "-K3"))


# --- as-of discipline -----------------------------------------------------------------------

def test_card_history_hides_the_present_and_future(be):
    row = be.get_transaction(FLAGGED)
    as_of = pd.Timestamp(row["ts"])
    h = be.card_history(row["card_key"], as_of=as_of)
    assert len(h) > 0
    assert (h.ts < as_of).all()
    assert FLAGGED not in set(h.TransactionID.astype(str))


def test_prior_cases_hide_cases_closed_later(be):
    cust = be.closed.customer_id.value_counts().index[0]
    all_cases = be.closed[be.closed.customer_id == cust].sort_values("closed_at")
    midpoint = all_cases.closed_at.iloc[len(all_cases) // 2]
    visible = be.prior_cases_for_customer(cust, as_of=midpoint)
    assert (visible.closed_at < midpoint).all()
    assert len(visible) < len(all_cases)


def test_similar_cases_hide_the_future(be):
    as_of = pd.Timestamp("2016-08-01")
    sim = be.similar_closed_cases(as_of=as_of, pattern="card_not_present_fraud", exposure_usd=200)
    assert len(sim) > 0
    assert (sim.closed_at < as_of).all()


def test_similar_cases_filter_structurally_before_ranking(be):
    as_of = pd.Timestamp("2016-11-01")
    sim = be.similar_closed_cases(as_of=as_of, pattern="out_of_region_use",
                                  exposure_usd=250, k=5)
    assert (sim.pattern == "out_of_region_use").all()
    # exposure band keeps the neighbours comparable rather than merely template-similar
    assert (sim.exposure_usd.between(100, 625)).all()


# --- behaviour ------------------------------------------------------------------------------

def test_baseline_matches_the_measured_triage(be):
    """HHG-006's amount sits at the 97th percentile of its own card's prior history."""
    row = be.get_transaction(FLAGGED)
    b = be.card_baseline(row["card_key"], FLAGGED)
    assert b.n_prior_txns == 101
    assert round(b.amt_pctile_prior, 2) == 0.97
    assert not b.new_region and not b.new_product


def test_baseline_is_comparable_across_wildly_different_volumes(be):
    """The same feature has to mean something for a 33-txn card and a 10,306-txn card."""
    for txn in (FLAGGED, "3583368"):          # HHG-006 and HHG-011
        row = be.get_transaction(txn)
        b = be.card_baseline(row["card_key"], txn)
        assert 0.0 <= b.amt_pctile_prior <= 1.0


def test_device_lookup_and_the_in_person_case(be):
    d = be.device_for_transaction(FLAGGED)
    assert d and d["device_state"] == "New"
    # HHG-001 is in-person (product W): no identity record is expected, not an error
    assert be.device_for_transaction("3514030") is None


def test_ring_signals_are_usually_empty_and_always_specific(be):
    row = be.get_transaction(FLAGGED)
    sigs = be.ring_signals(row["card_key"], as_of=pd.Timestamp(row["ts"]))
    for s in sigs:
        assert s.global_card_count <= 10       # the calibrated filter held
        assert s.n_cards >= 2
        assert s.span_days <= 14
        assert "device profile" in s.element   # R6 needs the element named


def test_card_testing_is_rare_and_time_scoped(be):
    """C11923 has an August burst and a December alert. The burst is not the December episode."""
    dec = be.card_testing_episode(
        be.resolve_card(txn_id="3583368").card_key, as_of=pd.Timestamp("2016-12-29"))
    assert dec is None


# --- case memory ----------------------------------------------------------------------------

def test_case_write_then_read_back(be, tmp_path):
    local = LocalBackend()
    local._cases_path = str(tmp_path / "cases.jsonl")
    gid = local.write_case({"case_id": "HHG-TEST", "verdict": "fraud",
                            "graph_case_id": "CASE-TEST-1"})
    assert gid == "CASE-TEST-1"
    got = local.read_case("CASE-TEST-1")
    assert got and got["verdict"] == "fraud"
    assert local.read_case("CASE-DOES-NOT-EXIST") is None


# --- R7 evidence: recurring charges ---------------------------------------------------------

def test_recurring_charge_returns_a_reasoned_answer_for_every_exam_case(be):
    """R7 is the only brake on R2, so this must answer - with a reason - on all 20.

    Not asserting which cases match: that is a finding, not a fixture. Asserting that the
    detector runs against real history, never claims a match without naming the cadence and the
    prior charges it found, and never cites a transaction at or after the flagged one.
    """
    pack = pd.read_parquet(os.path.join(DATA, "case_pack.parquet"))
    matched = []
    for row in pack.itertuples():
        card = be.resolve_card(txn_id=str(row.flagged_txn_id))
        assert card is not None, row.case_id
        r = be.recurring_charge(card.card_key, str(row.flagged_txn_id))
        assert r.reason or r.matches, f"{row.case_id}: no reason given"
        if r.matches:
            matched.append(row.case_id)
            assert r.cadence and r.interval_days > 0
            assert len(r.prior_txn_ids) >= 2
            flagged_ts = pd.Timestamp(be.get_transaction(str(row.flagged_txn_id))["ts"])
            for tid in r.prior_txn_ids:
                assert pd.Timestamp(be.get_transaction(tid)["ts"]) < flagged_ts
    # a detector that fires on every case is not a detector
    assert len(matched) < len(pack)
    print(f"\nrecurring match on {len(matched)}/{len(pack)}: {matched}")
