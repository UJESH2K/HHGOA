"""Reproduces every measured figure cited in README.md / PROJECT.md / ARCHITECTURE.md.

Usage:  python analysis/profile_dataset.py [--data DIR] [--out DIR]

Sections:
  A  file inventory + core distributions
  B  card_key reconstruction and its verification against closed cases
  C  device-profile rarity -> the ring-detection threshold
  D  card-testing detector (policy R5, literal reading) + threshold sensitivity
  E  closed cases as a labelled dev set (month split)
  F  cold-start triage features for the 20 exam cases  -> writes exam_triage.csv

Nothing here calls an LLM or a graph. Pure pandas, ~3 min on the full file.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]
CORE_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    *CARD_COLS, "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "customer_id", "ts", "channel", "risk_score",
]
DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "drive-download-20260919T105649Z-1-001")


def card_key(df: pd.DataFrame) -> pd.Series:
    """The verified card identifier. transactions.csv has no card_id column."""
    return df.customer_id + "::" + df[CARD_COLS].astype(str).agg("|".join, axis=1)


def device_profile(idf: pd.DataFrame) -> pd.Series:
    return (idf.DeviceInfo.fillna("?") + " | " + idf.id_30.fillna("?") + " | "
            + idf.id_31.fillna("?") + " | " + idf.id_33.fillna("?"))


def load(data_dir: str):
    txns = pd.read_csv(os.path.join(data_dir, "transactions.csv"), usecols=CORE_COLS)
    txns["ts"] = pd.to_datetime(txns.ts)
    txns["card_key"] = card_key(txns)
    ident = pd.read_csv(os.path.join(data_dir, "identity.csv"))
    ident["device_profile"] = device_profile(ident)
    ident["full_profile"] = (ident.DeviceInfo.notna() & ident.id_30.notna()
                             & ident.id_31.notna() & ident.id_33.notna())
    closed = pd.read_csv(os.path.join(data_dir, "closed_cases_history.csv"))
    closed["opened_dt"] = pd.to_datetime(closed.opened_at)
    pack = pd.read_csv(os.path.join(data_dir, "case_pack.csv"))
    return txns, ident, closed, pack


def section_a(txns, ident, closed, pack):
    print("\n" + "=" * 72, "\nA. INVENTORY")
    print(f"transactions {len(txns):,} rows | identity {len(ident):,} | closed {len(closed):,} | pack {len(pack)}")
    print(f"window {txns.ts.min()} -> {txns.ts.max()}")
    print(f"customers {txns.customer_id.nunique():,} | cards {txns.card_key.nunique():,}")
    print(f"channel {txns.channel.value_counts().to_dict()}")
    print(f"risk_score mean {txns.risk_score.mean():.3f} median {txns.risk_score.median():.2f} "
          f"| >0.7 {(txns.risk_score > 0.7).sum():,} | >=0.9 {(txns.risk_score >= 0.9).sum():,}")
    print(f"amount median ${txns.TransactionAmt.median():.2f} mean ${txns.TransactionAmt.mean():.2f} "
          f"max ${txns.TransactionAmt.max():,.2f}")
    exam = txns[txns.ts >= "2016-11-01"]
    print(f"exam window txns {len(exam):,} | of those >0.7 {(exam.risk_score > 0.7).sum():,}")
    print(f"addr1 null overall {txns.addr1.isna().sum():,}; by product "
          f"{txns.groupby('ProductCD').addr1.apply(lambda s: round(s.isna().mean(), 3)).to_dict()}")
    print(f"email domains (union P_/R_) {len(set(txns.P_emaildomain.dropna()) | set(txns.R_emaildomain.dropna()))}")


def section_b(txns, closed):
    print("\n" + "=" * 72, "\nB. card_key RECONSTRUCTION")
    key_of = txns.set_index("TransactionID").card_key
    single, multi, mapping = 0, 0, {}
    for _, r in closed.dropna(subset=["txn_ids"]).iterrows():
        ids = [int(x) for x in str(r.txn_ids).split("|") if x.strip()]
        keys = set(key_of.reindex(ids).dropna())
        single, multi = (single + 1, multi) if len(keys) == 1 else (single, multi + 1)
        for k in keys:
            mapping.setdefault(k, set()).add(r.card_id)
    ambiguous = sum(1 for v in mapping.values() if len(v) > 1)
    print(f"distinct card_keys {txns.card_key.nunique():,}")
    print(f"closed cases whose txns sit on ONE card_key: {single}/{single + multi} (multi: {multi})")
    print(f"card_keys mapping to >1 card_id: {ambiguous} of {len(mapping)}  <- must be 0")
    return mapping


def section_c(txns, ident):
    print("\n" + "=" * 72, "\nC. DEVICE-PROFILE RARITY -> RING THRESHOLD")
    j = ident[["TransactionID", "device_profile", "full_profile"]].merge(
        txns[["TransactionID", "card_key", "ts", "customer_id"]], on="TransactionID")
    g = j.groupby("device_profile").agg(cards=("card_key", "nunique"), full=("full_profile", "max"))
    print(f"profiles {len(g):,} | fully specified {int(g.full.sum()):,} | seen on >1 card {(g.cards > 1).sum():,}")
    for n in (1, 2, 3, 5, 10, 25, 50):
        print(f"  <= {n:>2} cards globally: {(g.cards <= n).sum():>5,} ({(g.cards <= n).mean():.1%})")
    print("full-profile card-count quantiles:", g[g.full].cards.quantile([.5, .75, .9, .95, .99]).round(1).to_dict())
    nd = j[(j.ts >= "2016-11-01") & j.full_profile]
    cand = nd.groupby("device_profile").agg(
        cards=("card_key", "nunique"), custs=("customer_id", "nunique"),
        first=("ts", "min"), last=("ts", "max"))
    cand["span_days"] = (cand["last"] - cand["first"]).dt.total_seconds() / 86400
    cand["global_cards"] = cand.index.map(g.cards)
    naive = j[j.ts >= "2016-11-01"].groupby("device_profile").customer_id.nunique()
    sel = cand[(cand.cards >= 2) & (cand.global_cards <= 10) & (cand.span_days <= 14)]
    print(f"naive 'shared in Nov-Dec' profiles: {(naive > 1).sum():,}")
    print(f"filtered (full + >=2 cards + <=10 global + <=14d): {len(sel):,}")
    print(sel.sort_values("cards", ascending=False).head(6).to_string())
    return sel


def section_d(txns, thresholds=(5, 10, 25)):
    print("\n" + "=" * 72, "\nD. CARD-TESTING DETECTOR (R5)")
    online = txns[txns.channel == "online"].sort_values(["card_key", "ts"])
    for thr in thresholds:
        small = online[online.TransactionAmt < thr]
        hits = []
        for k, grp in small.groupby("card_key"):
            if len(grp) < 3:
                continue
            ts = grp.ts.values
            for a in range(len(grp) - 2):
                b = a + 2
                while b < len(grp) and (ts[b] - ts[a]) / np.timedelta64(1, "h") <= 1.0:
                    b += 1
                if b - a >= 3:
                    end = grp.ts.iloc[b - 1]
                    later = txns[(txns.card_key == k) & (txns.ts > end)
                                 & (txns.ts <= end + pd.Timedelta(hours=24))
                                 & (txns.TransactionAmt > thr)]
                    hits.append((k, grp.ts.iloc[a], b - a, len(later),
                                 later.TransactionAmt.max() if len(later) else 0.0))
                    break
        ct = pd.DataFrame(hits, columns=["card_key", "window_start", "n_small", "n_follow", "max_follow_amt"])
        n_exam = (ct.window_start >= "2016-11-01").sum() if len(ct) else 0
        print(f"  amount < ${thr:>2}: {len(ct):>3} cards | with >${thr} follow-up {int((ct.n_follow > 0).sum()) if len(ct) else 0:>3}"
              f" | >$100 follow-up {int((ct.max_follow_amt > 100).sum()) if len(ct) else 0:>3} | in Nov-Dec {n_exam:>3}")
    return ct


def section_e(closed):
    print("\n" + "=" * 72, "\nE. CLOSED CASES AS DEV SET")
    closed["month"] = closed.opened_dt.dt.to_period("M").astype(str)
    print(pd.crosstab(closed.month, closed.outcome).to_string())
    print("\ncleared rate by month:",
          closed.groupby("month").outcome.apply(lambda s: round((s == "cleared").mean(), 3)).to_dict())
    hold = closed[closed.opened_dt >= "2016-10-01"]
    print(f"\nOct+ holdout: {len(hold)} cases, {(hold.outcome == 'confirmed_fraud').sum()} confirmed / "
          f"{(hold.outcome == 'cleared').sum()} cleared")
    print("holdout patterns:", hold.pattern.value_counts().to_dict())


def section_f(txns, ident, closed, pack, out_dir):
    print("\n" + "=" * 72, "\nF. EXAM COLD-START TRIAGE")
    flagged = txns[txns.TransactionID.isin(pack.flagged_txn_id)].set_index("TransactionID")
    idx = ident.set_index("TransactionID")
    rows = []
    for _, r in pack.iterrows():
        f = flagged.loc[r.flagged_txn_id]
        hist = txns[(txns.card_key == f.card_key) & (txns.ts < f.ts)]
        prior = closed[closed.customer_id == r.customer_id]
        rec = idx.loc[r.flagged_txn_id] if r.flagged_txn_id in idx.index else None
        rows.append(dict(
            case_id=r.case_id, trigger=r.trigger_type, amount=f.TransactionAmt,
            channel=f.channel, product=f.ProductCD, risk_score=f.risk_score,
            card_id=r.card_id, hist_txns=len(hist),
            amt_pctile=round((hist.TransactionAmt < f.TransactionAmt).mean(), 3) if len(hist) else None,
            new_region=bool(pd.notna(f.addr1) and len(hist) and f.addr1 not in set(hist.addr1.dropna())),
            new_product=bool(len(hist) and f.ProductCD not in set(hist.ProductCD)),
            id_15=(rec.id_15 if rec is not None and pd.notna(rec.id_15) else ""),
            proxy=(rec.id_23 if rec is not None and pd.notna(rec.id_23) else ""),
            prior_fraud=int((prior.outcome == "confirmed_fraud").sum()),
            prior_cleared=int((prior.outcome == "cleared").sum()),
            # --- consistency columns, the other half of the rubric ----------------------
            # Novelty raises suspicion; consistency lowers it. Both have to be observable or
            # the scorer can only ever escalate - see src/scoring/rubric.py, _profile_fit.
            # `known_*` is NOT the negation of `new_*`: a null value is neither.
            region_recorded=bool(pd.notna(f.addr1)),
            known_region=bool(pd.notna(f.addr1) and len(hist)
                              and f.addr1 in set(hist.addr1.dropna())),
            known_product=bool(len(hist) and f.ProductCD in set(hist.ProductCD)),
            new_email_domain=bool(pd.notna(f.P_emaildomain) and len(hist)
                                  and f.P_emaildomain not in set(hist.P_emaildomain.dropna())),
            known_email_domain=bool(pd.notna(f.P_emaildomain) and len(hist)
                                    and f.P_emaildomain in set(hist.P_emaildomain.dropna())),
            new_time_of_day=bool(len(hist) >= 20
                                 and (f.ts.hour // 6) not in set(hist.ts.dt.hour // 6)),
            timing_typical=bool(len(hist) >= 20
                                and (f.ts.hour // 6) in set(hist.ts.dt.hour // 6)),
            prior_1h=int(((hist.ts >= f.ts - pd.Timedelta(hours=1)) & (hist.ts < f.ts)).sum()),
            prior_24h=int(((hist.ts >= f.ts - pd.Timedelta(hours=24)) & (hist.ts < f.ts)).sum()),
        ))
    ex = pd.DataFrame(rows)
    print(ex.to_string(index=False))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "exam_triage.csv")
    ex.to_csv(path, index=False)
    print(f"\nwrote {path}")
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    txns, ident, closed, pack = load(a.data)
    section_a(txns, ident, closed, pack)
    section_b(txns, closed)
    section_c(txns, ident)
    section_d(txns)
    section_e(closed)
    section_f(txns, ident, closed, pack, a.out)


if __name__ == "__main__":
    main()
