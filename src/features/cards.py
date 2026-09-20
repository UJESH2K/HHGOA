"""Card identity: reconstruct the card, then resolve its real `card_id`.

`transactions.csv` has no card_id column, but `case_pack.csv` and `closed_cases_history.csv`
both key on one (`C01234-K1`). Two mechanisms, in strict precedence order:

  1. ANCHORED          - the case files hand us (card_id, txn_ids); map those txns to their
                         card_key. Ground truth. Covers ~1,956 card_keys.
  2. INFERRED_SINGLE   - a customer with exactly one card_key holds exactly one card, so that
                         card is "<customer_id>-K1". Verified on all 1,430 anchored cards whose
                         customer has a single key: 1,430/1,430 are K1, zero counterexamples.
                         Still an inference, so it is flagged as one.
  3. UNKNOWN           - a multi-card customer's card we have never seen named. The K-suffix is
                         NOT derivable (ordering by first_ts / last_ts / volume / min TransactionID
                         / card1 all fail to reproduce known suffixes - best was 60% on 2-card
                         customers, i.e. chance). These cards must never be cited by id in an
                         answer file; cite the customer and describe the link instead.

See PROJECT.md section 3 and `analysis/profile_dataset.py` section B.
"""
from __future__ import annotations

from enum import Enum

import pandas as pd

CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]


class Provenance(str, Enum):
    ANCHORED = "anchored"
    INFERRED_SINGLE = "inferred_single_card"
    UNKNOWN = "unknown"


def card_key(df: pd.DataFrame) -> pd.Series:
    """The verified card identifier: customer + the full six-column card tuple.

    0 of 1,956 keys reachable from closed cases map to more than one card_id.
    """
    return df.customer_id + "::" + df[CARD_COLS].astype(str).agg("|".join, axis=1)


def _anchors_from_closed(txns: pd.DataFrame, closed: pd.DataFrame) -> tuple[dict, list]:
    key_of = txns.set_index("TransactionID").card_key
    claims: dict[str, set] = {}
    for _, r in closed.dropna(subset=["txn_ids"]).iterrows():
        for raw in str(r.txn_ids).split("|"):
            raw = raw.strip()
            if not raw:
                continue
            k = key_of.get(int(raw))
            if k is not None:
                claims.setdefault(k, set()).add(r.card_id)
    conflicts = [(k, sorted(v)) for k, v in claims.items() if len(v) > 1]
    return {k: next(iter(v)) for k, v in claims.items() if len(v) == 1}, conflicts


def _anchors_from_pack(txns: pd.DataFrame, pack: pd.DataFrame) -> dict:
    flagged = txns.set_index("TransactionID").card_key
    out = {}
    for _, r in pack.iterrows():
        k = flagged.get(r.flagged_txn_id)
        if k is not None:
            out[k] = r.card_id
    return out


def resolve_card_ids(
    txns: pd.DataFrame,
    closed: pd.DataFrame | None = None,
    pack: pd.DataFrame | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Return one row per card_key: card_key, customer_id, card_id, provenance.

    `strict` raises if any card_key is claimed by two different card_ids, which would mean the
    card_key reconstruction is unsound. Measured: 0 conflicts.
    """
    if "card_key" not in txns.columns:
        txns = txns.assign(card_key=card_key(txns))

    anchors: dict[str, str] = {}
    conflicts: list = []
    if closed is not None:
        anchors, conflicts = _anchors_from_closed(txns, closed)
        if conflicts and strict:
            raise ValueError(
                f"card_key reconstruction is unsound: {len(conflicts)} key(s) map to multiple "
                f"card_ids, e.g. {conflicts[:3]}"
            )
    if pack is not None:
        # the exam pack is the most authoritative source; it wins any tie
        anchors.update(_anchors_from_pack(txns, pack))

    cards = (txns.groupby("card_key", as_index=False)
                 .agg(customer_id=("customer_id", "first"), n_txns=("TransactionID", "size")))
    n_keys = cards.groupby("customer_id").card_key.transform("size")

    cards["card_id"] = cards.card_key.map(anchors)
    cards["provenance"] = Provenance.UNKNOWN.value
    cards.loc[cards.card_id.notna(), "provenance"] = Provenance.ANCHORED.value

    single = cards.card_id.isna() & (n_keys == 1)
    cards.loc[single, "card_id"] = cards.loc[single, "customer_id"] + "-K1"
    cards.loc[single, "provenance"] = Provenance.INFERRED_SINGLE.value

    return cards[["card_key", "customer_id", "card_id", "provenance", "n_txns"]]


def citable_card_ids(cards: pd.DataFrame) -> set[str]:
    """card_ids an answer file is allowed to name. UNKNOWN cards are excluded by construction."""
    return set(cards.loc[cards.provenance != Provenance.UNKNOWN.value, "card_id"].dropna())
