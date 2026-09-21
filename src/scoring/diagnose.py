"""Why we do NOT fit a classifier on the closed cases.

    python -m src.scoring.diagnose

This module exists to keep a rejected approach honest and reproducible. The plan was to fit a
calibrated probability model on 5,565 labelled investigations - `fraud_probability` is scored
for calibration, and fitting to outcomes is how calibration is normally earned.

It does not work here, and the reason is worth carrying into the write-up.

FINDING. All 900 cleared cases carry `risk_score` >= 0.81; confirmed fraud spans 0.01-0.98.
Below 0.81 the history holds 0 legitimate and 4,038 fraud examples. A classifier therefore learns
`low score => fraud` - the inverse of how the score works - and posts a fake AUC of 0.991.

WHY IT HAPPENS. Not a corrupt file: it is what "closed case" means. A cleared case is an alert
that fired and was dismissed, and alerts fire on high scores. Confirmed fraud also arrives by
customer report, which carries a low score. The sampling is realistic; it just makes the set
unusable as training data for our population.

WHY IT MATTERS. 17 of the 20 exam cases sit below 0.81, where the training data has never seen a
legitimate outcome. A fitted model calls all 17 fraud with near-total confidence, against a brief
that says roughly half are legitimate.

CONSEQUENCE. `fraud_probability` comes from a transparent rubric (src/scoring/rubric.py) whose
weights are grounded in measured distributions rather than fitted to this label set. The closed
cases stay what the brief calls them: case memory for retrieval, and a qualitative backtest.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_FEATURES, extract

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
LEAK_THRESHOLD = 0.81


def anchor_txn_id(row) -> str | None:
    """The transaction a closed case was raised on.

    Confirmed fraud records `first_fraud_txn_id`; cleared cases leave it null but still list the
    alerted transaction in `txn_ids`. All 5,565 resolve.
    """
    if pd.notna(row.first_fraud_txn_id):
        return str(int(row.first_fraud_txn_id))
    if pd.notna(row.txn_ids):
        return str(row.txn_ids).split("|")[0].strip() or None
    return None


def build_training_set(data_dir: str = DATA) -> pd.DataFrame:
    """One as-of feature row per closed case. Kept for the backtest and this diagnosis."""
    txns = pd.read_parquet(os.path.join(data_dir, "txns.parquet"))
    txns["TransactionID"] = txns.TransactionID.astype(str)
    by_txn = txns.set_index("TransactionID", drop=False)

    ident = pd.read_parquet(os.path.join(data_dir, "identity.parquet"))
    ident["TransactionID"] = ident.TransactionID.astype(str)
    by_ident = ident.set_index("TransactionID", drop=False)

    profiles = pd.read_parquet(os.path.join(data_dir, "device_profiles.parquet"))
    prof_cards = dict(zip(profiles.device_profile, profiles.global_card_count))

    closed = pd.read_parquet(os.path.join(data_dir, "closed_cases.parquet"))
    closed["anchor"] = closed.apply(anchor_txn_id, axis=1)
    by_customer = {c: g.sort_values("closed_at") for c, g in closed.groupby("customer_id")}

    rows, meta = [], []
    for r in closed.itertuples():
        if r.anchor is None or r.anchor not in by_txn.index:
            continue
        txn = by_txn.loc[r.anchor]
        idrow = by_ident.loc[r.anchor] if r.anchor in by_ident.index else None
        hist = by_customer.get(r.customer_id)
        prior = hist[hist.closed_at < r.opened_at] if hist is not None else closed.iloc[:0]
        rows.append(extract(txn, idrow, prior,
                            prof_cards.get(idrow.device_profile) if idrow is not None else None))
        meta.append({"case_id": r.case_id, "opened_at": r.opened_at, "pattern": r.pattern,
                     "outcome": r.outcome, "label": 1 if r.outcome == "confirmed_fraud" else 0})
    df = pd.DataFrame(rows)
    for k in meta[0]:
        df[k] = [m[k] for m in meta]
    return df


def report(data_dir: str = DATA) -> dict:
    df = build_training_set(data_dir)
    cleared, fraud = df[df.label == 0], df[df.label == 1]

    print("=" * 72)
    print("1. THE LEAK")
    print(f"  cleared  risk_score: min {cleared.risk_score.min():.2f}  "
          f"max {cleared.risk_score.max():.2f}  (n={len(cleared)})")
    print(f"  fraud    risk_score: min {fraud.risk_score.min():.2f}  "
          f"max {fraud.risk_score.max():.2f}  (n={len(fraud)})")
    below = df[df.risk_score < LEAK_THRESHOLD]
    print(f"\n  below {LEAK_THRESHOLD}: {int((below.label == 0).sum())} legitimate, "
          f"{int(below.label.sum())} fraud  -> the label is decided by the score alone")

    print("\n2. WHAT A CLASSIFIER DOES WITH THAT")
    tr, te = df[df.opened_at < "2016-10-01"], df[df.opened_at >= "2016-10-01"]
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])
    pipe.fit(tr[ALL_FEATURES].to_numpy(float), tr.label.to_numpy())
    p = pipe.predict_proba(te[ALL_FEATURES].to_numpy(float))[:, 1]
    auc = roc_auc_score(te.label.to_numpy(), p)
    brier = brier_score_loss(te.label.to_numpy(), p)
    print(f"  held-out AUC {auc:.3f} | Brier {brier:.4f}   <- looks excellent, is worthless")
    rs_auc = roc_auc_score(df.label.to_numpy(), df.risk_score.to_numpy())
    print(f"  risk_score alone, INVERTED, gives AUC {1 - rs_auc:.3f} - that is the whole model")

    print("\n3. WHY IT CANNOT TRANSFER")
    pack = pd.read_parquet(os.path.join(data_dir, "case_pack.parquet"))
    txns = pd.read_parquet(os.path.join(data_dir, "txns.parquet"),
                           columns=["TransactionID", "risk_score"])
    txns["TransactionID"] = txns.TransactionID.astype(str)
    rs = dict(zip(txns.TransactionID, txns.risk_score))
    exam = pack.flagged_txn_id.astype(str).map(rs)
    n_below = int((exam < LEAK_THRESHOLD).sum())
    print(f"  exam cases below {LEAK_THRESHOLD}: {n_below} of {len(exam)}")
    print(f"  a model fitted above would call all {n_below} fraud with near-total confidence,")
    print("  against a brief that says roughly half the pack is legitimate.")

    print("\n4. THE ONLY COMPARABLE STRATUM")
    s = df[df.risk_score >= LEAK_THRESHOLD]
    print(f"  risk_score >= {LEAK_THRESHOLD}: n={len(s)}, {s.label.mean():.1%} fraud")
    rows = []
    for f in ALL_FEATURES:
        x = s[f].to_numpy(float)
        if np.std(x):
            a = roc_auc_score(s.label.to_numpy(), x)
            rows.append((f, a, abs(a - 0.5)))
    print("  strongest features there (several inverted - treat with suspicion):")
    for f, a, _ in sorted(rows, key=lambda r: -r[2])[:5]:
        print(f"    {f:<22} auc={a:.3f}")
    print(f"  ...and it covers only {int((exam >= LEAK_THRESHOLD).sum())} of the 20 exam cases.")

    print("\n" + "=" * 72)
    print("VERDICT: do not fit on closed cases. Use a transparent rubric; keep the closed")
    print("cases as retrieval memory and a qualitative backtest. See PROJECT.md section 3.")
    return {"auc_leaked": round(float(auc), 4), "exam_below_threshold": n_below,
            "cleared_min_risk": float(cleared.risk_score.min())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    report(ap.parse_args().data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
