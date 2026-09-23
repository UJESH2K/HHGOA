"""The two backends must answer the same questions the same way.

    TG_HOST=... TG_SECRET=... python -m pytest tests/test_tigergraph_contract.py -v

These ran green against a live Savanna workspace (engine 4.2.5) on 2026-09-23, which is what the
file was written for: the GSQL and the response parsing in `src/graph/tigergraph.py` were authored
without an endpoint to test against, and this is what turned the inevitable first-contact mistakes
into loud disagreements instead of quietly wrong answer files.

WHY EQUIVALENCE IS THE RIGHT TEST. TigerGraph is the default backend and the one demoed; the
parquet backend is the outage fallback and the backtest engine, because 5,565 closed-case replays
over the network are unaffordable. Those two roles are only safe if the agent genuinely cannot
tell them apart - otherwise the backtest measures one system and the submission ships another.

The tolerance on floats is deliberate and small. `amt_pctile_prior` is the feature that separates
the exam pack best, and the loader writes -1.0 where the source was null; a backend that returns
-1.0 as a percentile rather than None would put a case at the bottom of its own amount
distribution instead of saying "no prior history". That is a one-line parsing bug with a
several-case consequence, and `test_baseline_matches` is where it surfaces.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.graph.local import DATA, LocalBackend
from src.graph import tigergraph

# Which parquet build to compare the graph against. The graph holds whatever was last loaded, so
# the two have to be pointed at the same dataset: `CONTRACT_DATA=data_fixture` compares against
# the synthetic load, the default compares against the real one. Getting this wrong would fail
# every test for the honest reason that they are different datasets.
DATA_DIR = os.environ.get("CONTRACT_DATA") or DATA

pytestmark = [
    pytest.mark.skipif(not tigergraph.available(),
                       reason="no TG_HOST / pyTigerGraph - see STRATEGY.md section 9"),
    pytest.mark.skipif(not os.path.exists(os.path.join(DATA_DIR, "txns.parquet")),
                       reason=f"no parquet build in {DATA_DIR} - run src.features.build or "
                              "src.fixtures.generate"),
]

# Tolerance is per field, matched to what the field MEANS and the precision it is STORED at.
# A single global epsilon was wrong in both directions: too tight for dollar amounts, which
# `export.py` deliberately rounds to 4 decimals to keep the CSV small, and too loose to be worth
# having on the percentile, which is unit-scaled and is the feature that separates the exam pack
# best. A tenth of a cent on a mean does not change a decision; 1e-3 on a percentile could.
TOL = 1e-6
TOL_BY_FIELD = {
    "amt_mean_prior": 1e-3,     # dollars, rounded to 4dp on export
    "amt_max_prior": 1e-3,      # dollars, rounded to 2dp on export
    "secs_since_prev": 1e-1,    # seconds, rounded to 1dp on export
    "span_days": 1e-3,
}


def _tol_for(label: str) -> float:
    for field, tol in TOL_BY_FIELD.items():
        if field in label:
            return tol
    return TOL


@pytest.fixture(scope="module")
def backends():
    return LocalBackend(DATA_DIR), tigergraph.from_env()


@pytest.fixture(scope="module")
def cases(backends):
    """The exam pack, which is the population that actually matters."""
    local, _ = backends
    pack = pd.read_parquet(os.path.join(DATA_DIR, "case_pack.parquet"))
    rows = []
    for row in pack.itertuples():
        card = local.resolve_card(txn_id=str(row.flagged_txn_id))
        rows.append({"case_id": row.case_id, "txn_id": str(row.flagged_txn_id),
                     "customer_id": row.customer_id,
                     "card_key": card.card_key if card else "",
                     "as_of": pd.Timestamp(row.opened_at)})
    return rows


def _approx(a, b, label=""):
    if a is None or b is None:
        assert a == b, f"{label}: {a!r} vs {b!r} - one is None and the other is not"
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        tol = _tol_for(label)
        assert abs(float(a) - float(b)) <= tol, f"{label}: {a} vs {b} (tolerance {tol})"
    else:
        assert str(a) == str(b), f"{label}: {a!r} vs {b!r}"


# --- entity lookups -------------------------------------------------------------------------

def test_transaction_matches(backends, cases):
    local, tg = backends
    for c in cases:
        a, b = local.get_transaction(c["txn_id"]), tg.get_transaction(c["txn_id"])
        for field in ("TransactionAmt", "risk_score", "amt_pctile_prior", "prior_txns_1h",
                      "prior_txns_24h"):
            _approx(a.get(field), b.get(field), f"{c['case_id']}.{field}")
        for field in ("ProductCD", "channel", "card_key", "customer_id"):
            assert str(a.get(field)) == str(b.get(field)), f"{c['case_id']}.{field}"
        assert pd.Timestamp(a["ts"]) == pd.Timestamp(b["ts"]), c["case_id"]


def test_card_resolution_matches(backends, cases):
    """The card_id and its provenance both have to agree: a card whose K-suffix we never learned
    must not be citable on one backend and citable on the other."""
    local, tg = backends
    for c in cases:
        a = local.resolve_card(txn_id=c["txn_id"])
        b = tg.resolve_card(txn_id=c["txn_id"])
        assert (a is None) == (b is None), c["case_id"]
        if a:
            assert a.card_key == b.card_key, c["case_id"]
            assert a.card_id == b.card_id, c["case_id"]
            assert a.provenance == b.provenance, c["case_id"]
            assert a.citable == b.citable, c["case_id"]


def test_customer_card_count_matches(backends, cases):
    """R10 needs the customer to hold two cards. Disagreeing here means BLOCK_ALL_CARDS on one
    backend and not the other, on a permission the validator checks."""
    local, tg = backends
    for c in cases:
        assert (local.get_customer(c["customer_id"])["n_cards"]
                == tg.get_customer(c["customer_id"])["n_cards"]), c["case_id"]


# --- behaviour ------------------------------------------------------------------------------

def test_baseline_matches(backends, cases):
    local, tg = backends
    for c in cases:
        a = local.card_baseline(c["card_key"], c["txn_id"])
        b = tg.card_baseline(c["card_key"], c["txn_id"])
        _approx(a.amt_pctile_prior, b.amt_pctile_prior, f"{c['case_id']}.amt_pctile_prior")
        _approx(a.amt_mean_prior, b.amt_mean_prior, f"{c['case_id']}.amt_mean_prior")
        assert a.n_prior_txns == b.n_prior_txns, c["case_id"]
        assert a.new_region == b.new_region, c["case_id"]
        assert a.new_product == b.new_product, c["case_id"]
        assert a.prior_txns_1h == b.prior_txns_1h, c["case_id"]
        assert set(a.known_products) == set(b.known_products), c["case_id"]
        assert set(a.known_regions) == set(b.known_regions), c["case_id"]


def test_history_window_matches(backends, cases):
    local, tg = backends
    for c in cases:
        a = local.card_history(c["card_key"], c["as_of"], limit=10)
        b = tg.card_history(c["card_key"], c["as_of"], limit=10)
        assert list(a.TransactionID.astype(str)) == list(b.TransactionID.astype(str)), \
            c["case_id"]


def test_history_respects_the_cutoff(backends, cases):
    """The as-of rule, on both backends. A leak here inflates the backtest silently."""
    local, tg = backends
    for c in cases:
        for be in (local, tg):
            hist = be.card_history(c["card_key"], c["as_of"], limit=50)
            if len(hist):
                assert hist.ts.max() < c["as_of"], f"{c['case_id']} leaked post-cutoff rows"


def test_recurring_charge_matches(backends, cases):
    """R7's evidence. The arithmetic is shared code, so a disagreement means the graph returned a
    different candidate set - which is the interesting failure, not the boring one."""
    local, tg = backends
    for c in cases:
        a = local.recurring_charge(c["card_key"], c["txn_id"])
        b = tg.recurring_charge(c["card_key"], c["txn_id"])
        assert a.matches == b.matches, f"{c['case_id']}: {a.reason} vs {b.reason}"
        if a.matches:
            assert a.cadence == b.cadence, c["case_id"]
            assert set(a.prior_txn_ids) == set(b.prior_txn_ids), c["case_id"]


def test_device_matches(backends, cases):
    local, tg = backends
    for c in cases:
        a, b = local.device_for_transaction(c["txn_id"]), tg.device_for_transaction(c["txn_id"])
        assert (a is None) == (b is None), f"{c['case_id']}: one backend found a device"
        if a:
            assert a["device_profile"] == b["device_profile"], c["case_id"]
            assert a["device_state"] == b["device_state"], c["case_id"]
            assert a["proxy"] == b["proxy"], c["case_id"]
            assert a["is_full_profile"] == b["is_full_profile"], c["case_id"]


def test_ring_signals_match(backends, cases):
    """The expensive one to get wrong: under R6 a ring pulls in FILE_REPORT."""
    local, tg = backends
    for c in cases:
        a = local.ring_signals(c["card_key"], c["as_of"])
        b = tg.ring_signals(c["card_key"], c["as_of"])
        assert {r.device_profile for r in a} == {r.device_profile for r in b}, c["case_id"]
        for x, y in zip(sorted(a, key=lambda r: r.device_profile),
                        sorted(b, key=lambda r: r.device_profile)):
            assert x.n_cards == y.n_cards, c["case_id"]
            assert x.volume_artefact_risk == y.volume_artefact_risk, c["case_id"]


def test_card_testing_matches(backends, cases):
    local, tg = backends
    for c in cases:
        a = local.card_testing_episode(c["card_key"], c["as_of"])
        b = tg.card_testing_episode(c["card_key"], c["as_of"])
        assert (a is None) == (b is None), c["case_id"]
        if a:
            assert set(a["small_txn_ids"]) == set(b["small_txn_ids"]), c["case_id"]


# --- case memory ----------------------------------------------------------------------------

def test_prior_cases_match(backends, cases):
    local, tg = backends
    for c in cases:
        a = local.prior_cases_for_customer(c["customer_id"], c["as_of"])
        b = tg.prior_cases_for_customer(c["customer_id"], c["as_of"])
        assert set(a.case_id) == set(b.case_id), c["case_id"]


def test_prior_cases_respect_the_cutoff(backends, cases):
    """In the backtest this is the difference between an investigation and a lookup: a case
    opened after the one being replayed would hand over the answer."""
    local, tg = backends
    for c in cases:
        for be in (local, tg):
            prior = be.prior_cases_for_customer(c["customer_id"], c["as_of"])
            if len(prior):
                assert prior.closed_at.max() < c["as_of"], c["case_id"]


def test_case_write_and_read_back(backends):
    """`written_to_graph` in an answer file is set by this round trip, never by a constant."""
    _, tg = backends
    record = {"case_id": "HHG-CONTRACT-TEST", "graph_case_id": "CASE-CONTRACT-TEST",
              "customer_id": "", "verdict": "uncertain", "fraud_probability": 0.42,
              "pattern": "none", "exposure_usd": 0.0, "status": "open",
              "summary": "contract test", "actions": ["MONITOR_CARD"],
              "evidence": [{"claim": "x", "source": "graph", "ref": "test", "entity_ids": []}],
              "decisions": ["written by the contract test"]}
    gid = tg.write_case(record)
    back = tg.read_case(gid)
    assert back is not None, "the case was written but cannot be read back"
    assert back["verdict"] == "uncertain"
    assert back["actions"] == ["MONITOR_CARD"]
    assert back["evidence"] and back["evidence"][0]["source"] == "graph"
