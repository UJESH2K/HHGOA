"""Phase 0b pipeline: raw CSV -> data/*.parquet with every derived feature.

    python -m src.features.build

Runs once, takes a few minutes, and is the input to everything downstream: the local graph
backend, the backtest, and the TigerGraph loading jobs. Nothing else should ever re-parse the
675 MB CSV.

Exit gate (PLAN.md Phase 0b) is asserted at the end: 13,553 customers / 14,893 cards /
590,742 txns / 144,432 identity / 5,565 closed cases / 0 card_id conflicts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from . import baseline, rings, testing
from .cards import CARD_COLS, Provenance, card_key, resolve_card_ids

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_IN = os.path.join(ROOT, "drive-download-20260919T105649Z-1-001")
DATA_OUT = os.path.join(ROOT, "data")

CORE_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", *CARD_COLS,
    "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain",
    "customer_id", "ts", "channel", "risk_score",
]

EXPECTED = {"customers": 13553, "cards": 14893, "txns": 590742,
            "identity": 144432, "closed_cases": 5565}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(data_in: str = DATA_IN, data_out: str = DATA_OUT) -> dict:
    os.makedirs(data_out, exist_ok=True)

    log("reading transactions.csv (675 MB, core columns only)")
    txns = pd.read_csv(os.path.join(data_in, "transactions.csv"), usecols=CORE_COLS)
    txns["ts"] = pd.to_datetime(txns.ts)
    txns["card_key"] = card_key(txns)
    log(f"  {len(txns):,} transactions, {txns.card_key.nunique():,} card keys")

    log("reading identity.csv / closed_cases_history.csv / case_pack.csv")
    ident = pd.read_csv(os.path.join(data_in, "identity.csv"))
    closed = pd.read_csv(os.path.join(data_in, "closed_cases_history.csv"))
    closed["opened_at"] = pd.to_datetime(closed.opened_at)
    closed["closed_at"] = pd.to_datetime(closed.closed_at)
    pack = pd.read_csv(os.path.join(data_in, "case_pack.csv"))
    pack["opened_at"] = pd.to_datetime(pack.opened_at)

    log("resolving card_id (anchor from case files, infer for single-card customers)")
    cards = resolve_card_ids(txns, closed=closed, pack=pack, strict=True)
    prov = cards.provenance.value_counts().to_dict()
    log(f"  {prov}")
    txns = txns.merge(cards[["card_key", "card_id", "provenance"]], on="card_key", how="left")

    log("building sequence + amount baseline + novelty + velocity (as-of)")
    txns = baseline.build(txns)

    log("building device profiles")
    profiles = rings.build_profiles(ident, txns)
    ident = ident.assign(device_profile=rings.device_profile(ident))
    ident["is_full"] = ident[rings.PROFILE_FIELDS].notna().all(axis=1)
    specific = int(rings.is_specific(profiles).sum())
    log(f"  {len(profiles):,} profiles, {specific:,} specific enough to link cards")

    log("running R5 card-testing detector")
    ct = testing.detect(txns)
    log(f"  {len(ct)} cards with a qualifying burst, {int(ct.escalate.sum()) if len(ct) else 0} escalating")

    log("writing parquet")
    txns.to_parquet(os.path.join(data_out, "txns.parquet"), index=False)
    ident.to_parquet(os.path.join(data_out, "identity.parquet"), index=False)
    closed.to_parquet(os.path.join(data_out, "closed_cases.parquet"), index=False)
    pack.to_parquet(os.path.join(data_out, "case_pack.parquet"), index=False)
    cards.to_parquet(os.path.join(data_out, "cards.parquet"), index=False)
    profiles.to_parquet(os.path.join(data_out, "device_profiles.parquet"), index=False)
    if len(ct):
        ct.assign(small_txn_ids=ct.small_txn_ids.astype(str),
                  follow_txn_ids=ct.follow_txn_ids.astype(str)
                  ).to_parquet(os.path.join(data_out, "card_testing.parquet"), index=False)

    counts = {"customers": txns.customer_id.nunique(), "cards": len(cards),
              "txns": len(txns), "identity": len(ident), "closed_cases": len(closed)}
    return {"counts": counts, "provenance": prov, "card_testing_hits": len(ct),
            "specific_profiles": specific}


def check_gate(counts: dict) -> bool:
    ok = True
    for k, want in EXPECTED.items():
        got = counts[k]
        flag = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {k:<13} {got:>9,}  expected {want:>9,}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-in", default=DATA_IN)
    ap.add_argument("--data-out", default=DATA_OUT)
    a = ap.parse_args()

    t0 = time.time()
    result = run(a.data_in, a.data_out)
    print("\nPhase 0b exit gate (PLAN.md):")
    ok = check_gate(result["counts"])
    print(f"\ndone in {time.time() - t0:.0f}s -> {a.data_out}")
    if not ok:
        print("\nGATE FAILED - the load is wrong. Stop and fix before Phase 1.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
