"""Feature extraction for the calibrated fraud scorer.

One function, used by BOTH training and inference, so the model can never be fed a
differently-shaped vector at run time than it was fitted on.

Every feature is as-of by construction: the transaction-level ones were computed with an
expanding window in `features/baseline.py`, and the case-history ones filter on `closed_at <
as_of` here. A closed case must not be able to see itself or anything that happened later.

Deliberately excluded:
  - `exposure_usd`, `n_txns`, `actions_taken`, `report_filed` - all recorded at case CLOSE.
    Using them would leak the outcome and produce a scorer that looks perfect and predicts nothing.
  - raw `card_id` / `customer_id` - identity, not behaviour; would memorise the 1,892 customers
    who happen to appear in history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "log_amount",
    "amt_pctile_prior",
    "amt_vs_mean_prior",
    "amt_vs_max_prior",
    "risk_score",
    "is_online",
    "log_prior_txns",
    "new_region",
    "new_product",
    "prior_txns_1h",
    "prior_txns_24h",
    "log_secs_since_prev",
    "device_new",
    "device_found",
    "device_missing",
    "has_proxy",
    "profile_is_full",
    "profile_rare",
    "prior_fraud_cases",
    "prior_cleared_cases",
    "has_prior_history",
]

# product codes as one-hot; W is the in-person baseline and stays implicit
PRODUCTS = ["C", "H", "R", "S"]
ALL_FEATURES = FEATURE_NAMES + [f"product_{p}" for p in PRODUCTS]


def _safe_log(x, floor=1e-3):
    return float(np.log10(max(float(x), floor) + 1.0))


def extract(txn: pd.Series, identity: pd.Series | None, prior_cases: pd.DataFrame,
            profile_card_count: int | None = None) -> dict[str, float]:
    """Feature vector for one transaction viewed as an alert.

    `txn`          - a row of txns.parquet (carries the as-of baselines already)
    `identity`     - the matching identity row, or None for in-person
    `prior_cases`  - closed cases for this customer with closed_at < as_of
    """
    amt = float(txn.TransactionAmt)
    mean_prior = txn.get("amt_mean_prior")
    max_prior = txn.get("amt_max_prior")

    f: dict[str, float] = {
        "log_amount": _safe_log(amt),
        # a first transaction has no prior distribution; 0.5 is the neutral prior
        "amt_pctile_prior": 0.5 if pd.isna(txn.get("amt_pctile_prior")) else float(txn.amt_pctile_prior),
        "amt_vs_mean_prior": (amt / float(mean_prior)) if pd.notna(mean_prior) and mean_prior else 1.0,
        "amt_vs_max_prior": (amt / float(max_prior)) if pd.notna(max_prior) and max_prior else 1.0,
        "risk_score": float(txn.risk_score),
        "is_online": 1.0 if txn.channel == "online" else 0.0,
        "log_prior_txns": _safe_log(txn.get("txn_seq", 0)),
        "new_region": 1.0 if bool(txn.get("new_region", False)) else 0.0,
        "new_product": 1.0 if bool(txn.get("new_product", False)) else 0.0,
        "prior_txns_1h": float(txn.get("prior_txns_1h", 0)),
        "prior_txns_24h": float(txn.get("prior_txns_24h", 0)),
        "log_secs_since_prev": _safe_log(txn.get("secs_since_prev") if pd.notna(
            txn.get("secs_since_prev")) else 86400 * 30),
    }

    # clip the ratios: a $400 charge on a card whose prior mean is $0.50 is informative,
    # but 800.0 as a raw feature value dominates a linear model
    f["amt_vs_mean_prior"] = min(f["amt_vs_mean_prior"], 20.0)
    f["amt_vs_max_prior"] = min(f["amt_vs_max_prior"], 10.0)

    # device. id_15=New covers 43% of all identity records, so it is a weak feature by
    # construction - included so the model can learn exactly how weak, rather than assumed.
    state = None if identity is None else identity.get("id_15")
    f["device_new"] = 1.0 if state == "New" else 0.0
    f["device_found"] = 1.0 if state == "Found" else 0.0
    f["device_missing"] = 1.0 if identity is None else 0.0
    f["has_proxy"] = 1.0 if (identity is not None and pd.notna(identity.get("id_23"))) else 0.0
    f["profile_is_full"] = 1.0 if (identity is not None and bool(identity.get("is_full"))) else 0.0
    f["profile_rare"] = 1.0 if (profile_card_count is not None and profile_card_count <= 10) else 0.0

    n_fraud = int((prior_cases.outcome == "confirmed_fraud").sum()) if len(prior_cases) else 0
    n_clear = int((prior_cases.outcome == "cleared").sum()) if len(prior_cases) else 0
    f["prior_fraud_cases"] = float(min(n_fraud, 20))
    f["prior_cleared_cases"] = float(min(n_clear, 20))
    f["has_prior_history"] = 1.0 if (n_fraud + n_clear) else 0.0

    for p in PRODUCTS:
        f[f"product_{p}"] = 1.0 if txn.ProductCD == p else 0.0

    return f


def to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    return np.array([[r[name] for name in ALL_FEATURES] for r in rows], dtype=float)
