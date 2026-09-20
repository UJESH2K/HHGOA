"""Per-card behavioural baselines, computed as-of so no case can see its own future.

The single most useful feature in the build is `amt_pctile_prior`: where this transaction's
amount sits in the distribution of everything that card did BEFORE it. It means the same thing
for a 36-transaction card and a 10,306-transaction card, which raw counts and raw amounts do not
(PROJECT.md section 3 - the exam pack spans both).

Every column here is strictly backward-looking. A feature that peeks at later transactions would
inflate the closed-case backtest and teach us nothing about the 20.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_sequence(txns: pd.DataFrame) -> pd.DataFrame:
    """The NEXT chain and time deltas, per card, ordered by ts. Load-time job, not query-time."""
    t = txns.sort_values(["card_key", "ts"]).copy()
    g = t.groupby("card_key", sort=False)
    t["prev_txn_id"] = g.TransactionID.shift(1)
    t["next_txn_id"] = g.TransactionID.shift(-1)
    t["secs_since_prev"] = g.ts.diff().dt.total_seconds()
    t["txn_seq"] = g.cumcount()  # 0-based position in this card's history
    return t


def add_amount_baseline(txns: pd.DataFrame) -> pd.DataFrame:
    """`amt_pctile_prior` = share of this card's EARLIER transactions with a smaller amount.

    NaN for a card's first transaction (no prior distribution to sit in).

    Implementation note: expanding rank with method="min" gives the 1-based rank of the current
    value among the first n values, counting ties at the lowest rank. So (rank_min - 1) is exactly
    the count of strictly-smaller earlier values, and dividing by txn_seq gives the share.
    """
    t = txns if "txn_seq" in txns.columns else add_sequence(txns)
    rank_min = (t.groupby("card_key", sort=False).TransactionAmt
                 .expanding().rank(method="min")
                 .reset_index(level=0, drop=True))
    seq = t["txn_seq"]
    t = t.copy()
    t["amt_pctile_prior"] = np.where(seq > 0, (rank_min.to_numpy() - 1.0) / seq.replace(0, np.nan), np.nan)

    # rolling context the agent reasons with, all backward-looking
    g = t.groupby("card_key", sort=False).TransactionAmt
    t["amt_mean_prior"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    t["amt_max_prior"] = g.apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)
    return t


def add_novelty(txns: pd.DataFrame) -> pd.DataFrame:
    """Has this card used this billing region / product code before?

    A null addr1 yields False, not True - 94.8% of `C`-product rows have no region at all, and
    "no region recorded" is not "a new region" (PROJECT.md caution 6).
    """
    t = txns.copy()
    for col, flag in (("addr1", "new_region"), ("ProductCD", "new_product")):
        seen_before = (t.groupby(["card_key", col], sort=False).cumcount() > 0)
        first_use = ~seen_before
        has_value = t[col].notna()
        # novel only if this is the card's first use of a value it actually has, and the card
        # has prior history at all
        t[flag] = first_use & has_value & (t["txn_seq"] > 0)
    return t


def add_velocity(txns: pd.DataFrame, windows_h=(1, 24)) -> pd.DataFrame:
    """Count of prior transactions on this card within the trailing window. Burst detection."""
    t = txns.copy()
    for h in windows_h:
        counts = np.zeros(len(t), dtype=int)
        pos = 0
        for _, grp in t.groupby("card_key", sort=False):
            ts = grp.ts.to_numpy()
            n = len(ts)
            # for each i, how many j < i fall inside the trailing window
            left = np.searchsorted(ts, ts - np.timedelta64(h, "h"), side="left")
            counts[pos:pos + n] = np.arange(n) - left
            pos += n
        t[f"prior_txns_{h}h"] = counts
    return t


def build(txns: pd.DataFrame) -> pd.DataFrame:
    return add_velocity(add_novelty(add_amount_baseline(add_sequence(txns))))
