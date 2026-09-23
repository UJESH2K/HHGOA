"""Policy R5 card-testing detector - deterministic, exact, and cheap.

R5 is precise enough to implement literally: three or more small online authorizations on one
card within an hour, followed by a larger purchase. No LLM judgement required, so this runs on
every case regardless of what the model suspects.

Measured across the whole book (analysis/profile_dataset.py section D):

    threshold   cards   with follow-up   follow-up > $100   in Nov-Dec
    < $5           28              19                  6            9
    < $10          93              53                 29           18
    < $25         343             149                 83           61

No exam case has a qualifying sequence at its flagged moment, at any threshold. Closed-case
history agrees that this is rare: 16 of 5,565. Cite the detector when it fires; treat a
card_testing verdict without it as a claim needing unusually strong support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMALL_AMOUNT = 5.0      # R5's "often under $5"
WINDOW_HOURS = 1.0      # R5's "within an hour"
FOLLOW_HOURS = 24.0     # how long after the burst we look for the larger purchase
ESCALATE_ABOVE = 100.0  # R5: "if a purchase over $100 has already cleared, recommend BLOCK_CARD"
LOOKBACK_DAYS = 7       # how far back a burst still counts as THIS episode (see episode_for_card)


def detect(txns: pd.DataFrame, small_amount: float = SMALL_AMOUNT,
           window_hours: float = WINDOW_HOURS) -> pd.DataFrame:
    """One row per card with a qualifying testing burst.

    Columns: card_key, window_start, window_end, n_small, small_txn_ids, n_follow,
             max_follow_amt, follow_txn_ids, escalate (a >$100 purchase already cleared).
    """
    online = txns[txns.channel == "online"].sort_values(["card_key", "ts"])
    small = online[online.TransactionAmt < small_amount]
    rows = []

    for key, grp in small.groupby("card_key", sort=False):
        if len(grp) < 3:
            continue
        ts = grp.ts.to_numpy()
        # widest burst starting at each position; take the first that reaches 3
        for a in range(len(grp) - 2):
            b = a + 2
            while b < len(grp) and (ts[b] - ts[a]) / np.timedelta64(1, "h") <= window_hours:
                b += 1
            if b - a < 3:
                continue
            burst = grp.iloc[a:b]
            end = burst.ts.iloc[-1]
            card_txns = txns[txns.card_key == key]
            follow = card_txns[(card_txns.ts > end)
                               & (card_txns.ts <= end + pd.Timedelta(hours=FOLLOW_HOURS))
                               & (card_txns.TransactionAmt > small_amount)]
            rows.append(dict(
                card_key=key,
                window_start=burst.ts.iloc[0], window_end=end,
                n_small=len(burst),
                small_txn_ids=burst.TransactionID.tolist(),
                n_follow=len(follow),
                max_follow_amt=float(follow.TransactionAmt.max()) if len(follow) else 0.0,
                follow_txn_ids=follow.TransactionID.tolist(),
            ))
            break  # one burst per card is enough to raise the pattern

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["card_key", "window_start", "window_end", "n_small",
                                     "small_txn_ids", "n_follow", "max_follow_amt",
                                     "follow_txn_ids", "escalate"])
    out["escalate"] = out.max_follow_amt > ESCALATE_ABOVE
    return out.sort_values("window_start").reset_index(drop=True)


def episode_for_card(hits: pd.DataFrame, card_key: str, at: pd.Timestamp | None = None,
                     within_days: int = LOOKBACK_DAYS) -> dict | None:
    """The testing episode relevant to a card at a point in time, or None.

    `at` bounds relevance: C11923 has a burst in August and an alert in December. That burst is
    prior history worth citing, not the current episode - so it is not returned for a December
    investigation.

    The window looks BACKWARDS ONLY (`at - within_days <= window_start < at`). A burst that
    starts after the flagged moment is the investigation's own future, and every read in this
    codebase is as-of. `graph/local.card_testing_episode` uses the same bounds; the two must not
    drift apart, because one is the backtest engine and the other is the graph query.
    """
    if hits.empty:
        return None
    m = hits[hits.card_key == card_key]
    if at is not None:
        m = m[(m.window_start >= at - pd.Timedelta(days=within_days))
              & (m.window_start < at)]
    if m.empty:
        return None
    r = m.iloc[0]
    # ONE SHAPE FOR BOTH BACKENDS. This used to return `txn_ids` while the parquet backend
    # returned `small_txn_ids` / `follow_txn_ids`, so the agent - which reads the latter - would
    # have raised KeyError the first time a card-testing case ran through TigerGraph. The
    # contract test found it; the fix is that there is only one shape.
    small = [str(i) for i in r.small_txn_ids]
    follow = [str(i) for i in r.follow_txn_ids]
    return dict(card_key=r.card_key,
                small_txn_ids=small, follow_txn_ids=follow,
                txn_ids=small + follow,
                n_small=int(r.n_small), max_follow_amt=float(r.max_follow_amt),
                escalate=bool(r.escalate),
                window=[str(r.window_start), str(r.window_end)])
