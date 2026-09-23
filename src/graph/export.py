"""Turn the parquet build into the CSVs the TigerGraph loading jobs expect.

    python -m src.graph.export --data data --out graph_csv
    python -m src.graph.export --data data --out graph_csv --subgraph

WHAT THIS ADDS BEYOND A COLUMN RENAME. Three of the outputs do not exist anywhere until this
module builds them, and each of them is a design decision from ARCHITECTURE.md made concrete:

  `next_edges.csv`          the NEXT chain, ordered by ts within a card. Load-time job, not a
                            query-time sort - it is what makes burst and sequence evidence one
                            hop instead of a full-history scan.
  `shares_device_edges.csv` the FILTERED shared-device links. The calibration lives in the edge
                            set: only profiles that are fully specified and appear on at most ten
                            cards book-wide produce an edge at all. Without that filter,
                            connected components over this edge type returns one component
                            holding half the book - 4,793 of 9,706 profiles span more than one
                            card and the largest is 1,023 cards of all-null fields.
  `involves_edges.csv`      the closed-case to transaction links, exploded from the pipe-joined
                            `txn_ids` column.

THE SUBGRAPH CUT. `--subgraph` writes every card involved in the 20 exam cases and the 5,565
closed cases at FULL history, plus the entire Nov-Dec window, and nothing else. It exists because
590,742 transaction vertices plus ~590k NEXT edges may not fit a free workspace, and because the
alternative - sampling - would make graded traversals wrong in ways that are invisible. Every
query the investigation runs stays exact on this cut; what is lost is unrelated history, and the
manifest records exactly what was dropped.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from ..features import rings

NOV_DEC_START = "2016-11-01"


def _dt(series: pd.Series) -> pd.Series:
    """TigerGraph's DATETIME wants `YYYY-MM-DD HH:MM:SS` with no fractional part."""
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d %H:%M:%S")


def _blank(series: pd.Series) -> pd.Series:
    """Nulls become empty strings, because the loading jobs test `!= ""` to skip an edge.

    A null billing region is not a region called "nan": 94.8% of C-product transactions have no
    region at all, and writing the string "nan" into 62,000 BillingRegion vertices would create a
    fake hub that every ring query would then have to special-case.
    """
    return series.where(series.notna(), "").astype(str).replace({"nan": "", "None": ""})


def select_subgraph(txns: pd.DataFrame, closed: pd.DataFrame,
                    pack: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The documented cut: case-involved cards at full history, plus all of Nov-Dec."""
    keep_keys = set(txns.loc[txns.TransactionID.astype(str).isin(
        pack.flagged_txn_id.astype(str)), "card_key"])

    involved: set[str] = set()
    for raw in closed.txn_ids.dropna().astype(str):
        involved.update(t.strip() for t in raw.split("|") if t.strip())
    keep_keys |= set(txns.loc[txns.TransactionID.astype(str).isin(involved), "card_key"])

    in_window = txns.ts >= NOV_DEC_START
    on_kept_card = txns.card_key.isin(keep_keys)
    cut = txns[in_window | on_kept_card].copy()

    note = {
        "mode": "subgraph",
        "cards_at_full_history": len(keep_keys),
        "txns_kept": int(len(cut)),
        "txns_dropped": int(len(txns) - len(cut)),
        "rule": ("every card involved in the exam pack or a closed case, at full history, plus "
                 f"every transaction from {NOV_DEC_START} onwards"),
        "why": ("a free workspace may not hold 590,742 transaction vertices and ~590k NEXT "
                "edges; sampling would make graded traversals wrong invisibly, so the cut is "
                "by relevance and is recorded here"),
    }
    return cut, note


def build_next_edges(txns: pd.DataFrame) -> pd.DataFrame:
    """The NEXT chain from the precomputed `next_txn_id` column."""
    chain = txns.loc[txns.next_txn_id.notna(), ["TransactionID", "next_txn_id", "card_key", "ts"]]
    out = pd.DataFrame({
        "from_txn_id": chain.TransactionID.astype(str),
        "to_txn_id": chain.next_txn_id.astype(str).str.replace(r"\.0$", "", regex=True),
    })
    # keep only edges whose target survived a subgraph cut, or the loader creates orphan vertices
    alive = set(txns.TransactionID.astype(str))
    out = out[out.to_txn_id.isin(alive)]
    gaps = txns.set_index(txns.TransactionID.astype(str)).secs_since_prev
    out["secs_gap"] = out.to_txn_id.map(gaps).fillna(0.0).round(1)
    return out


def build_shares_device_edges(txns: pd.DataFrame, ident: pd.DataFrame,
                              profiles: pd.DataFrame) -> pd.DataFrame:
    """Filtered card-to-card shared-device links, one row per (card pair, profile).

    The filter is `rings.is_specific`: fully specified profile, at most ten cards book-wide. Cards
    belonging to the SAME customer are excluded - a household with two cards on one laptop is not
    a ring, and leaving those edges in would make every multi-card customer look like one.
    """
    specific = set(profiles.loc[rings.is_specific(profiles), "device_profile"])
    j = ident[ident.device_profile.isin(specific)][["TransactionID", "device_profile"]]
    j = j.merge(txns[["TransactionID", "card_key", "customer_id", "ts"]], on="TransactionID")
    if j.empty:
        return pd.DataFrame(columns=["card_key_a", "card_key_b", "device_profile", "first_ts",
                                     "last_ts", "span_days", "n_customers"])

    rows = []
    for profile, grp in j.groupby("device_profile"):
        per_card = grp.groupby("card_key").agg(first_ts=("ts", "min"), last_ts=("ts", "max"),
                                               customer_id=("customer_id", "first"))
        keys = sorted(per_card.index)
        if len(keys) < 2:
            continue
        for a_i in range(len(keys)):
            for b_i in range(a_i + 1, len(keys)):
                a, b = keys[a_i], keys[b_i]
                if per_card.loc[a, "customer_id"] == per_card.loc[b, "customer_id"]:
                    continue                      # same cardholder, not a shared origin
                first = min(per_card.loc[a, "first_ts"], per_card.loc[b, "first_ts"])
                last = max(per_card.loc[a, "last_ts"], per_card.loc[b, "last_ts"])
                rows.append({
                    "card_key_a": a, "card_key_b": b, "device_profile": profile,
                    "first_ts": first, "last_ts": last,
                    "span_days": round((last - first).total_seconds() / 86400, 3),
                    "n_customers": int(grp.customer_id.nunique()),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["first_ts"] = _dt(out.first_ts)
        out["last_ts"] = _dt(out.last_ts)
    return out


def build_involves_edges(closed: pd.DataFrame, alive: set[str]) -> pd.DataFrame:
    rows = []
    for row in closed.itertuples():
        for raw in str(getattr(row, "txn_ids", "") or "").split("|"):
            tid = raw.strip()
            if tid and tid in alive:
                rows.append({"case_id": row.case_id, "txn_id": tid})
    return pd.DataFrame(rows, columns=["case_id", "txn_id"])


def run(data_dir: str, out_dir: str, subgraph: bool = False) -> dict:
    if not os.path.exists(os.path.join(data_dir, "txns.parquet")):
        raise SystemExit("\n".join([
            "",
            f"No parquet build in {data_dir}/",
            "",
            "  for the real dataset:  python -m src.features.build",
            "  for the fixture:       python -m src.fixtures.generate",
            "                         then pass --data data_fixture",
            "",
        ]))
    os.makedirs(out_dir, exist_ok=True)
    txns = pd.read_parquet(os.path.join(data_dir, "txns.parquet"))
    ident = pd.read_parquet(os.path.join(data_dir, "identity.parquet"))
    cards = pd.read_parquet(os.path.join(data_dir, "cards.parquet"))
    profiles = pd.read_parquet(os.path.join(data_dir, "device_profiles.parquet"))
    closed = pd.read_parquet(os.path.join(data_dir, "closed_cases.parquet"))
    pack = pd.read_parquet(os.path.join(data_dir, "case_pack.parquet"))

    note = {"mode": "full", "txns_kept": int(len(txns)), "txns_dropped": 0}
    if subgraph:
        txns, note = select_subgraph(txns, closed, pack)
        ident = ident[ident.TransactionID.isin(txns.TransactionID)]

    alive = set(txns.TransactionID.astype(str))

    # --- vertices ---------------------------------------------------------------------------
    customers = (txns.groupby("customer_id")
                 .agg(n_txns=("TransactionID", "size"), first_seen=("ts", "min"),
                      last_seen=("ts", "max")).reset_index())
    customers = customers.merge(
        cards.groupby("customer_id").size().rename("n_cards").reset_index(),
        on="customer_id", how="left")
    customers["n_cards"] = customers.n_cards.fillna(0).astype(int)
    customers["first_seen"] = _dt(customers.first_seen)
    customers["last_seen"] = _dt(customers.last_seen)
    customers = customers[["customer_id", "n_cards", "n_txns", "first_seen", "last_seen"]]

    card_meta = (txns.groupby("card_key")
                 .agg(network=("card4", "first"), card_type=("card6", "first")).reset_index())
    cards_out = cards.merge(card_meta, on="card_key", how="left")
    cards_out["card_id"] = _blank(cards_out.card_id)
    cards_out["network"] = _blank(cards_out.network)
    cards_out["card_type"] = _blank(cards_out.card_type)
    cards_out = cards_out[["card_key", "card_id", "customer_id", "provenance", "n_txns",
                           "network", "card_type"]]

    t = pd.DataFrame({
        "txn_id": txns.TransactionID.astype(str),
        "ts": _dt(txns.ts),
        "amount": txns.TransactionAmt.round(2),
        "product_cd": _blank(txns.ProductCD),
        "channel": _blank(txns.channel),
        "risk_score": txns.risk_score,
        "addr1": _blank(txns.addr1).str.replace(r"\.0$", "", regex=True),
        "p_emaildomain": _blank(txns.P_emaildomain),
        "card_key": txns.card_key,
        "customer_id": txns.customer_id,
        "amt_pctile_prior": txns.amt_pctile_prior.fillna(-1.0).round(6),
        "amt_mean_prior": txns.amt_mean_prior.fillna(-1.0).round(4),
        "amt_max_prior": txns.amt_max_prior.fillna(-1.0).round(2),
        "new_region": txns.new_region.astype(int),
        "new_product": txns.new_product.astype(int),
        "prior_txns_1h": txns.prior_txns_1h,
        "prior_txns_24h": txns.prior_txns_24h,
        "secs_since_prev": txns.secs_since_prev.fillna(-1.0).round(1),
        "txn_seq": txns.txn_seq,
        "next_txn_id": _blank(txns.next_txn_id).str.replace(r"\.0$", "", regex=True),
    })

    profiles_out = profiles.copy()
    profiles_out["is_full"] = profiles_out.is_full.astype(int)
    profiles_out["first_seen"] = _dt(profiles_out.first_seen)
    profiles_out["last_seen"] = _dt(profiles_out.last_seen)
    profiles_out = profiles_out[["device_profile", "is_full", "global_card_count",
                                 "global_customer_count", "n_txns", "first_seen", "last_seen"]]

    regions = (t[t.addr1 != ""].groupby("addr1").size().rename("n_txns").reset_index())
    domains = (t[t.p_emaildomain != ""].groupby("p_emaildomain").size()
               .rename("n_txns").reset_index().rename(columns={"p_emaildomain": "domain"}))

    # closed cases: attach the card_key so CASE_ON_CARD can be loaded
    key_of = txns.set_index(txns.TransactionID.astype(str)).card_key
    closed_out = closed.copy()
    closed_out["anchor"] = closed_out.get(
        "anchor_txn_id",
        closed_out.txn_ids.astype(str).str.split("|").str[0].str.strip())
    closed_out["card_key"] = closed_out.anchor.astype(str).map(key_of).fillna("")
    closed_out["opened_at"] = _dt(closed_out.opened_at)
    closed_out["closed_at"] = _dt(closed_out.closed_at)
    for col in ("channel", "product_cd", "actions_taken", "report_filed", "pattern",
                "analyst_notes", "card_id"):
        closed_out[col] = _blank(closed_out.get(col, pd.Series("", index=closed_out.index)))
    closed_out = closed_out[["case_id", "customer_id", "card_id", "opened_at", "closed_at",
                             "outcome", "pattern", "exposure_usd", "n_txns", "actions_taken",
                             "report_filed", "channel", "product_cd", "analyst_notes",
                             "card_key"]]

    on_device = ident[["TransactionID", "device_profile"]].copy()
    on_device["txn_id"] = on_device.TransactionID.astype(str)
    on_device["device_state"] = _blank(ident.get("id_15", pd.Series(index=ident.index)))
    on_device["proxy"] = _blank(ident.get("id_23", pd.Series(index=ident.index)))
    on_device = on_device[on_device.txn_id.isin(alive)][
        ["txn_id", "device_profile", "device_state", "proxy"]]

    outputs = {
        "customers.csv": customers,
        "cards.csv": cards_out,
        "transactions.csv": t,
        "device_profiles.csv": profiles_out,
        "billing_regions.csv": regions,
        "email_domains.csv": domains,
        "closed_cases.csv": closed_out,
        "next_edges.csv": build_next_edges(txns),
        "on_device_edges.csv": on_device,
        "shares_device_edges.csv": build_shares_device_edges(txns, ident, profiles),
        "involves_edges.csv": build_involves_edges(closed, alive),
    }

    counts = {}
    for name, df in outputs.items():
        df.to_csv(os.path.join(out_dir, name), index=False)
        counts[name] = int(len(df))

    manifest = {"data_dir": os.path.abspath(data_dir), "out_dir": os.path.abspath(out_dir),
                "selection": note, "rows": counts}
    with open(os.path.join(out_dir, "_export_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="graph_csv")
    ap.add_argument("--subgraph", action="store_true",
                    help="write the documented relevance cut instead of the whole book")
    a = ap.parse_args()

    manifest = run(a.data, a.out, a.subgraph)
    print(f"export -> {a.out}  ({manifest['selection']['mode']})")
    for name, n in manifest["rows"].items():
        print(f"  {name:<26} {n:>9,}")
    if manifest["selection"]["txns_dropped"]:
        print(f"\n  dropped {manifest['selection']['txns_dropped']:,} transactions "
              f"({manifest['selection']['rule']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
