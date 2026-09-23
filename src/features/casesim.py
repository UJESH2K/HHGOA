"""Ranking retrieved prior cases - one implementation, shared by both backends.

WHY THIS IS PYTHON AND NOT GSQL. The first version computed the similarity score inside the
GSQL query with an accumulator over an embedding passed as a `LIST<DOUBLE>` parameter. The engine
refuses that outright:

    Type Check Error (TYP-1001): The identifier 'query_embedding' of type list parameter
    is invalid to call any function

A `LIST` query parameter cannot be iterated or have `.size()` called on it, so neither the
`FOREACH` nor the length guard was legal, and the query saved as a DRAFT that could not be
installed - which surfaced later as a 404 on the REST endpoint rather than as an install error.

So the split is the same one used for the recurring-charge and card-testing detectors: the graph
does the retrieval and the filtering, which is what it is good at, and the scoring arithmetic
lives here, once, where both backends call it. That is also what lets the contract test prove the
two agree - a formula implemented twice is a formula that will diverge.

True vector ranking is not lost: `gsql/vector_upgrade.gsql` does it natively with TigerVector's
`vectorSearch`, which takes the query vector properly because it is a built-in rather than user
code iterating a parameter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Weights sum to 1.0. Each is a structural claim about what makes a prior case comparable, and
# the closed-case history is templated prose, so these carry more signal than text similarity.
W_PATTERN = 0.35
W_CHANNEL = 0.20
W_EXPOSURE = 0.30
W_RECENCY = 0.15

RECENCY_HORIZON_DAYS = 180.0
"""The book is six months long, so a case at the far end of it scores zero on recency."""


def exposure_closeness(exposures: pd.Series, target: float) -> pd.Series:
    """1.0 at the same exposure, decaying to 0 an order of magnitude away.

    Log-scaled because the history spans $0.27 to $35,031: a fixed dollar window is the entire
    distribution at one end and nothing at the other.
    """
    if not target or target <= 0:
        return pd.Series(0.0, index=exposures.index)
    ratio = exposures.clip(lower=0.01) / target
    return (1.0 - np.log10(ratio).abs().clip(0, 1)).astype(float)


def score(pool: pd.DataFrame, *, as_of: pd.Timestamp, channel: str | None = None,
          pattern: str | None = None, exposure_usd: float | None = None,
          k: int = 5) -> pd.DataFrame:
    """Add a `similarity` column to already-filtered candidates and return the top k.

    `pool` must already be restricted to cases visible at `as_of` - the caller owns the as-of cut
    because the caller is the one holding the query. Scoring a case that should not be visible
    would be a leak wherever the filtering happened.
    """
    if pool is None or pool.empty:
        out = pool if pool is not None else pd.DataFrame()
        return out.assign(similarity=pd.Series(dtype=float))

    scored = pool.copy()
    sim = pd.Series(0.0, index=scored.index)

    if pattern and "pattern" in scored.columns:
        sim += W_PATTERN * (scored.pattern == pattern)
    if channel and "channel" in scored.columns:
        sim += W_CHANNEL * (scored.channel == channel)
    if exposure_usd is not None and exposure_usd > 0 and "exposure_usd" in scored.columns:
        sim += W_EXPOSURE * exposure_closeness(scored.exposure_usd, exposure_usd)
    if "closed_at" in scored.columns:
        age_days = (pd.Timestamp(as_of) - pd.to_datetime(scored.closed_at)).dt.total_seconds() / 86400
        sim += W_RECENCY * (1.0 - (age_days / RECENCY_HORIZON_DAYS).clip(0, 1))

    scored["similarity"] = sim.round(4)
    sort_cols = ["similarity"] + (["closed_at"] if "closed_at" in scored.columns else [])
    return scored.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(k)


def narrow(pool: pd.DataFrame, *, channel: str | None = None, pattern: str | None = None,
           exposure_usd: float | None = None, k: int = 5) -> pd.DataFrame:
    """Apply the structural filters, relaxing any that would leave nothing to rank.

    The order is deliberate: pattern, then channel, then exposure band. Each filter is only kept
    if at least `k` candidates survive it, because a perfectly-filtered set of one is worse
    evidence than a loosely-filtered set of five - and `similar_prior_cases` is a scored field.
    """
    staged = pool
    if pattern is not None and pattern != "" and "pattern" in staged.columns:
        narrowed = staged[staged.pattern == pattern]
        if len(narrowed) >= k:
            staged = narrowed
    if channel and "channel" in staged.columns:
        narrowed = staged[staged.channel == channel]
        if len(narrowed) >= k:
            staged = narrowed
    if exposure_usd is not None and exposure_usd > 0 and "exposure_usd" in staged.columns:
        lo, hi = exposure_usd * 0.4, exposure_usd * 2.5
        narrowed = staged[(staged.exposure_usd >= lo) & (staged.exposure_usd <= hi)]
        if len(narrowed) >= k:
            staged = narrowed
    return staged
