"""Parquet/pandas implementation of GraphBackend.

Fast enough to replay thousands of closed cases, and it keeps the deliverable reachable if
TigerGraph is unavailable. Loads ~590k rows into memory once (~200 MB) and indexes them.

Cases written here go to `data/cases_written.jsonl` so case memory is testable without a graph.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from ..features import casesim, recurring, rings, testing
from ..features.recurring import Recurrence
from .interface import Baseline, CardSummary, GraphBackend, RingSignal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


class LocalBackend(GraphBackend):
    def __init__(self, data_dir: str = DATA):
        self.data_dir = data_dir
        self.txns = pd.read_parquet(os.path.join(data_dir, "txns.parquet"))
        self.txns["TransactionID"] = self.txns.TransactionID.astype(str)
        self._by_txn = self.txns.set_index("TransactionID", drop=False)
        self._by_card = {k: g for k, g in self.txns.groupby("card_key", sort=False)}

        self.identity = pd.read_parquet(os.path.join(data_dir, "identity.parquet"))
        self.identity["TransactionID"] = self.identity.TransactionID.astype(str)
        self._ident_by_txn = self.identity.set_index("TransactionID", drop=False)

        self.closed = pd.read_parquet(os.path.join(data_dir, "closed_cases.parquet"))
        self.cards = pd.read_parquet(os.path.join(data_dir, "cards.parquet"))
        self._card_by_key = self.cards.set_index("card_key", drop=False)
        self._card_by_id = (self.cards.dropna(subset=["card_id"])
                                .drop_duplicates("card_id").set_index("card_id", drop=False))
        self.profiles = pd.read_parquet(os.path.join(data_dir, "device_profiles.parquet"))

        ct_path = os.path.join(data_dir, "card_testing.parquet")
        self.card_testing = pd.read_parquet(ct_path) if os.path.exists(ct_path) else pd.DataFrame()
        self._cases_path = os.path.join(data_dir, "cases_written.jsonl")

    # --- entity lookups ----------------------------------------------------------------------

    def get_transaction(self, txn_id: str) -> dict[str, Any]:
        row = self._by_txn.loc[str(txn_id)]
        return json_safe(row.to_dict())

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        cards = self.customer_cards(customer_id)
        t = self.txns[self.txns.customer_id == customer_id]
        return {"customer_id": customer_id, "n_cards": len(cards), "n_txns": len(t),
                "first_seen": str(t.ts.min()) if len(t) else None,
                "last_seen": str(t.ts.max()) if len(t) else None,
                "cards": [c.__dict__ for c in cards]}

    def customer_cards(self, customer_id: str) -> list[CardSummary]:
        rows = self.cards[self.cards.customer_id == customer_id]
        return [CardSummary(card_id=r.card_id if pd.notna(r.card_id) else None,
                            card_key=r.card_key, customer_id=r.customer_id,
                            n_txns=int(r.n_txns), provenance=r.provenance)
                for r in rows.itertuples()]

    def resolve_card(self, *, card_id=None, card_key=None, txn_id=None) -> CardSummary | None:
        if txn_id is not None:
            card_key = self._by_txn.loc[str(txn_id), "card_key"]
        if card_key is not None:
            if card_key not in self._card_by_key.index:
                return None
            r = self._card_by_key.loc[card_key]
        elif card_id is not None:
            if card_id not in self._card_by_id.index:
                return None
            r = self._card_by_id.loc[card_id]
        else:
            return None
        g = self._by_card.get(r.card_key)
        return CardSummary(card_id=r.card_id if pd.notna(r.card_id) else None,
                           card_key=r.card_key, customer_id=r.customer_id,
                           n_txns=int(r.n_txns), provenance=r.provenance,
                           first_seen=str(g.ts.min()) if g is not None else None,
                           last_seen=str(g.ts.max()) if g is not None else None)

    # --- behaviour ---------------------------------------------------------------------------

    def card_history(self, card_key: str, as_of: pd.Timestamp, limit: int = 25,
                     days: int | None = None) -> pd.DataFrame:
        g = self._by_card.get(card_key)
        if g is None:
            return self.txns.iloc[:0]
        h = g[g.ts < as_of]
        if days is not None:
            h = h[h.ts >= as_of - pd.Timedelta(days=days)]
        cols = ["TransactionID", "ts", "TransactionAmt", "ProductCD", "channel", "addr1",
                "P_emaildomain", "risk_score", "amt_pctile_prior", "secs_since_prev"]
        return h.sort_values("ts").tail(limit)[cols]

    def card_baseline(self, card_key: str, txn_id: str) -> Baseline:
        row = self._by_txn.loc[str(txn_id)]
        g = self._by_card.get(card_key, self.txns.iloc[:0])
        prior = g[g.ts < row.ts]
        return Baseline(
            n_prior_txns=len(prior),
            amt_pctile_prior=none_if_nan(row.amt_pctile_prior),
            amt_mean_prior=none_if_nan(row.amt_mean_prior),
            amt_max_prior=none_if_nan(row.amt_max_prior),
            new_region=bool(row.new_region), new_product=bool(row.new_product),
            prior_txns_1h=int(row.prior_txns_1h), prior_txns_24h=int(row.prior_txns_24h),
            # `addr1` is a float column in pandas, so str() gives "204.0" for region 204. That
            # is not a number, it is a code - and it was reaching the answer files as
            # "Billing region 204.0 has never appeared on this card". The graph stores it
            # normalised, so the contract test caught the disagreement.
            known_regions=sorted({_region_code(x) for x in prior.addr1.dropna().unique()}),
            known_products=sorted(prior.ProductCD.dropna().unique().tolist()))

    def recurring_charge(self, card_key: str, txn_id: str) -> Recurrence:
        row = self._by_txn.loc[str(txn_id)]
        g = self._by_card.get(card_key)
        if g is None:
            return Recurrence(False, reason="The card has no prior history to establish a "
                                            "pattern in")
        return recurring.detect(
            g, amount=float(row.TransactionAmt), ts=row.ts,
            product_cd=row.ProductCD if pd.notna(row.ProductCD) else None,
            email_domain=row.P_emaildomain if pd.notna(row.P_emaildomain) else None,
            n_prior_txns=int((g.ts < row.ts).sum()))

    # --- device / ring -----------------------------------------------------------------------

    def device_for_transaction(self, txn_id: str) -> dict[str, Any] | None:
        tid = str(txn_id)
        if tid not in self._ident_by_txn.index:
            return None      # in-person (product W) has no identity record, and that is normal
        r = self._ident_by_txn.loc[tid]
        prof = self.profiles[self.profiles.device_profile == r.device_profile]
        return {"device_profile": r.device_profile, "DeviceType": none_if_nan(r.DeviceType),
                "DeviceInfo": none_if_nan(r.DeviceInfo), "os": none_if_nan(r.id_30),
                "browser": none_if_nan(r.id_31), "screen": none_if_nan(r.id_33),
                "device_state": none_if_nan(r.id_15),       # New / Found / Unknown - 43% are New
                "proxy": none_if_nan(r.id_23),
                "is_full_profile": bool(r.is_full),
                "global_card_count": int(prof.global_card_count.iloc[0]) if len(prof) else None}

    def ring_signals(self, card_key: str, as_of: pd.Timestamp,
                     window_days: int = 14) -> list[RingSignal]:
        g = self._by_card.get(card_key)
        if g is None:
            return []
        window_start = as_of - pd.Timedelta(days=window_days)
        mine = g[(g.ts >= window_start) & (g.ts < as_of)]
        my_profiles = set(self._ident_by_txn.reindex(
            mine.TransactionID).device_profile.dropna())
        if not my_profiles:
            return []

        specific = set(self.profiles.loc[rings.is_specific(self.profiles), "device_profile"])
        my_profiles &= specific
        if not my_profiles:
            return []

        j = self._ident_by_txn[self._ident_by_txn.device_profile.isin(my_profiles)]
        j = self.txns[self.txns.TransactionID.isin(j.TransactionID)]
        j = j[(j.ts >= window_start) & (j.ts < as_of)]
        # reset_index: `_ident_by_txn` keeps TransactionID as BOTH index and column, and
        # merging on an ambiguous key raises rather than picking one.
        j = j.merge(self._ident_by_txn[["TransactionID", "device_profile"]]
                    .reset_index(drop=True), on="TransactionID")

        out: list[RingSignal] = []
        for prof, grp in j.groupby("device_profile"):
            keys = sorted(set(grp.card_key))
            if len(keys) < rings.MIN_CARDS_IN_WINDOW:
                continue
            span = (grp.ts.max() - grp.ts.min()).total_seconds() / 86400
            if span > window_days:
                continue
            vols = [len(self._by_card.get(k, [])) for k in keys]
            ids = [self._card_by_key.loc[k, "card_id"] for k in keys]
            global_cards = int(self.profiles.loc[
                self.profiles.device_profile == prof, "global_card_count"].iloc[0])
            strength = rings.link_strength(global_cards)
            n_customers = grp.customer_id.nunique()
            # A moderate-strength profile is a common-ish device model; two cards on one is a
            # coincidence. Three unrelated cardholders inside the window is a pattern.
            if strength == "moderate" and n_customers < rings.MIN_CUSTOMERS_FOR_MODERATE:
                continue
            if strength == "none":
                continue
            out.append(RingSignal(
                device_profile=prof, card_keys=keys,
                card_ids=[i for i in ids if isinstance(i, str)],
                n_cards=len(keys), n_customers=n_customers, span_days=round(span, 2),
                global_card_count=global_cards, strength=strength,
                # a card with thousands of transactions touches many devices; if EVERY card on
                # the profile is high-volume this is probably an artefact, not a ring
                volume_artefact_risk=min(vols) >= 1000))
        return sorted(out, key=lambda r: (-r.n_cards, r.span_days))

    # --- case memory -------------------------------------------------------------------------

    def prior_cases_for_customer(self, customer_id: str, as_of: pd.Timestamp) -> pd.DataFrame:
        c = self.closed[(self.closed.customer_id == customer_id)
                        & (self.closed.closed_at < as_of)]
        return c.sort_values("opened_at")

    def similar_closed_cases(self, *, as_of, channel=None, pattern=None, exposure_usd=None,
                             k: int = 5) -> pd.DataFrame:
        """Structural filter, then the shared scorer in `features/casesim.py`.

        Both backends call the same two functions on the same columns, which is what makes the
        contract test able to prove they agree rather than merely look similar.
        """
        pool = self.closed[self.closed.closed_at < as_of]
        staged = casesim.narrow(pool, channel=channel, pattern=pattern,
                                exposure_usd=exposure_usd, k=k)
        return casesim.score(staged, as_of=as_of, channel=channel, pattern=pattern,
                             exposure_usd=exposure_usd, k=k)

    def card_testing_episode(self, card_key: str, as_of: pd.Timestamp) -> dict | None:
        if self.card_testing.empty:
            return None
        m = self.card_testing[self.card_testing.card_key == card_key]
        if m.empty:
            return None
        m = m[(m.window_start < as_of)
              & (m.window_start >= as_of - pd.Timedelta(days=testing.LOOKBACK_DAYS))]
        if m.empty:
            return None
        r = m.iloc[0]
        return {"card_key": r.card_key, "n_small": int(r.n_small),
                "max_follow_amt": float(r.max_follow_amt), "escalate": bool(r.escalate),
                "window": [str(r.window_start), str(r.window_end)],
                "small_txn_ids": _id_list(r.small_txn_ids),
                "follow_txn_ids": _id_list(r.follow_txn_ids)}

    # --- writes ------------------------------------------------------------------------------

    def write_case(self, case: dict) -> str:
        graph_case_id = case.get("graph_case_id") or f"CASE-LOCAL-{case.get('case_id', 'X')}"
        record = {**case, "graph_case_id": graph_case_id}
        with open(self._cases_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return graph_case_id

    def read_case(self, graph_case_id: str) -> dict | None:
        if not os.path.exists(self._cases_path):
            return None
        with open(self._cases_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("graph_case_id") == graph_case_id:
                    return rec
        return None


def _id_list(v) -> list[str]:
    """Transaction-id columns must come back as lists of strings, whatever parquet returns."""
    if v is None or isinstance(v, float):
        return []
    if isinstance(v, str):                      # tolerate parquet files written before the fix
        return [t.strip().strip("'\"") for t in v.strip("[]").split(",") if t.strip()]
    return [str(x) for x in v]


def _region_code(v) -> str:
    """Region codes as codes: 204.0 -> "204". Shared with `graph/export.py`, which does the
    same normalisation on the way into TigerGraph."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v)


def none_if_nan(v):
    return None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)


def json_safe(d: dict) -> dict:
    return {k: none_if_nan(v) if not isinstance(v, (list, dict)) else v for k, v in d.items()}
