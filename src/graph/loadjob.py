"""Generate the loading job from the exported CSVs, so the two cannot drift apart.

    python -m src.graph.loadjob --csv graph_csv_fixture        # print it
    python -m src.graph.loadjob --csv graph_csv --out /tmp/x   # write it

WHY GENERATED RATHER THAN HAND-WRITTEN. The hand-written version referenced columns by name
(`$"customer_id"`), which the engine rejects unless the FILENAME is initialised with a real
server-side path - and on Savanna every path we can name is inside a directory the engine refuses
as sensitive. The alternative the engine offers is positional columns (`$0`, `$1`), which work
everywhere and are unreadable and fragile: a column inserted in `export.py` silently shifts every
index after it and the load succeeds with the wrong data in the wrong attributes.

So the mapping is declared once here, by column NAME, and the positions are resolved by reading
the header row of the actual CSV that will be uploaded. Rename or reorder a column in `export.py`
and this either follows it or fails loudly with the name it could not find. Nothing silently
loads into the wrong field.
"""
from __future__ import annotations

import argparse
import csv
import os

# target -> (csv file, kind, target name, [column names in the target's attribute order])
# The column ORDER here must match the attribute order in schema.gsql, because that is what
# `VALUES (...)` is positional against. The names are checked against the CSV header.
SKIP = "_"
"""GSQL's skip-this-attribute token, for columns a later pass fills in (the embeddings)."""

MAPPINGS: list[tuple[str, str, str, str, list[str]]] = [
    # (tag, csv, kind, target, columns)
    ("f_customers", "customers.csv", "VERTEX", "Customer",
     ["customer_id", "n_cards", "n_txns", "first_seen", "last_seen"]),
    ("f_cards", "cards.csv", "VERTEX", "PaymentCard",
     ["card_key", "card_id", "customer_id", "provenance", "n_txns", "network", "card_type"]),
    ("f_txns", "transactions.csv", "VERTEX", "Transaction",
     ["txn_id", "ts", "amount", "product_cd", "channel", "risk_score", "addr1",
      "p_emaildomain", "card_key", "customer_id", "amt_pctile_prior", "amt_mean_prior",
      "amt_max_prior", "new_region", "new_product", "prior_txns_1h", "prior_txns_24h",
      "secs_since_prev", "txn_seq", "next_txn_id"]),
    ("f_devices", "device_profiles.csv", "VERTEX", "DeviceProfile",
     ["device_profile", "is_full", "global_card_count", "global_customer_count", "n_txns",
      "first_seen", "last_seen"]),
    ("f_regions", "billing_regions.csv", "VERTEX", "BillingRegion", ["addr1", "n_txns"]),
    ("f_domains", "email_domains.csv", "VERTEX", "EmailDomain", ["domain", "n_txns"]),
    ("f_closed", "closed_cases.csv", "VERTEX", "ClosedCase",
     ["case_id", "customer_id", "card_id", "opened_at", "closed_at", "outcome", "pattern",
      "exposure_usd", "n_txns", "actions_taken", "report_filed", "channel", "product_cd",
      "analyst_notes", SKIP]),   # SKIP = summary_embedding, loaded by the embedding pass

    ("f_cards", "cards.csv", "EDGE", "OWNS", ["customer_id", "card_key"]),
    ("f_txns", "transactions.csv", "EDGE", "USED_IN", ["card_key", "txn_id"]),
    ("f_next", "next_edges.csv", "EDGE", "NEXT_TXN",
     ["from_txn_id", "to_txn_id", "secs_gap"]),
    ("f_on_device", "on_device_edges.csv", "EDGE", "ON_DEVICE",
     ["txn_id", "device_profile", "device_state", "proxy"]),
    ("f_shares", "shares_device_edges.csv", "EDGE", "SHARES_DEVICE",
     ["card_key_a", "card_key_b", "device_profile", "first_ts", "last_ts", "span_days",
      "n_customers"]),
    ("f_closed", "closed_cases.csv", "EDGE", "HAS_CASE", ["customer_id", "case_id"]),
    ("f_involves", "involves_edges.csv", "EDGE", "INVOLVES", ["case_id", "txn_id"]),
]

# Edges whose source or target may be blank in the CSV; a blank key would create a junk vertex.
SKIP_IF_BLANK = {
    "BILLED_TO": "addr1",
    "FROM_DOMAIN": "p_emaildomain",
    "CASE_ON_CARD": "card_key",
}

# These two hang off the transaction/closed-case rows and are conditional, so they are declared
# separately rather than squeezed into the table above.
CONDITIONAL: list[tuple[str, str, str, str, list[str], str]] = [
    ("f_txns", "transactions.csv", "EDGE", "BILLED_TO", ["txn_id", "addr1"], "addr1"),
    ("f_txns", "transactions.csv", "EDGE", "FROM_DOMAIN", ["txn_id", "p_emaildomain"],
     "p_emaildomain"),
    ("f_closed", "closed_cases.csv", "EDGE", "CASE_ON_CARD", ["case_id", "card_key"], "card_key"),
]


def header_of(csv_dir: str, name: str) -> list[str]:
    path = os.path.join(csv_dir, name)
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def positions(header: list[str], columns: list[str], where: str) -> list[int | str]:
    out: list[int | str] = []
    for col in columns:
        if col == SKIP:
            out.append(SKIP)
            continue
        if col not in header:
            raise KeyError(
                f"{where}: column {col!r} is not in the exported CSV header {header}. "
                "Either export.py stopped writing it or it was renamed - fix the mapping in "
                "src/graph/loadjob.py rather than guessing a position.")
        out.append(header.index(col))
    return out


def generate(csv_dir: str, graph: str, job: str = "load_fraud_graph") -> str:
    tags = sorted({tag for tag, *_ in MAPPINGS} | {tag for tag, *_ in CONDITIONAL})
    present = {name for name in os.listdir(csv_dir) if name.endswith(".csv")}

    lines = [
        "// GENERATED by `python -m src.graph.loadjob` - do not edit by hand.",
        "//",
        "// Positions are resolved from the header row of the CSVs in " + os.path.basename(csv_dir),
        "// so that a column added or reordered in export.py cannot silently load into the wrong",
        "// attribute. The mapping is declared by NAME in src/graph/loadjob.py.",
        "//",
        "// No header option: files POSTed over REST are parsed from line 1, so deploy.py strips",
        "// the header row before upload. With header=\"true\" the engine ignored it and loaded",
        "// the header as a data row - every vertex type came out exactly one too many.",
        "",
        f"USE GRAPH {graph}",
        "",
        f"CREATE LOADING JOB {job} FOR GRAPH {graph} {{",
    ]
    for tag in tags:
        lines.append(f"  DEFINE FILENAME {tag};")
    lines.append("")

    def emit(tag, csv_name, kind, target, columns, skip_col=None):
        if csv_name not in present:
            lines.append(f"  // skipped {target}: {csv_name} was not exported")
            return
        header = header_of(csv_dir, csv_name)
        idx = positions(header, columns, f"{kind} {target}")
        values = ", ".join(SKIP if i == SKIP else f"${i}" for i in idx)
        named = ", ".join(f"{c}=${i}" if i != SKIP else f"{c}=(skipped)"
                          for c, i in zip(columns, idx))
        lines.append(f"  // {named}")
        stmt = f"  LOAD {tag} TO {kind} {target} VALUES ({values})"
        if skip_col:
            stmt += f' WHERE ${header.index(skip_col)} != ""'
        stmt += ' USING SEPARATOR=",", QUOTE="double";'
        lines.append(stmt)

    lines.append("  // --- vertices ---")
    for tag, csv_name, kind, target, columns in MAPPINGS:
        if kind == "VERTEX":
            emit(tag, csv_name, kind, target, columns)
    lines.append("")
    lines.append("  // --- edges ---")
    for tag, csv_name, kind, target, columns in MAPPINGS:
        if kind == "EDGE":
            emit(tag, csv_name, kind, target, columns)
    for tag, csv_name, kind, target, columns, skip in CONDITIONAL:
        emit(tag, csv_name, kind, target, columns, skip)

    lines.append("}")
    # A job with no LOAD statements is not an empty job, it is a GSQL syntax error: the engine
    # reports `Encountered "}"` at the closing brace, which says nothing about the real cause
    # (the CSV directory is empty because export never ran). Refuse here, where the cause is
    # still visible.
    if not any(line.lstrip().startswith("LOAD ") for line in lines):
        raise SystemExit("\n".join([
            "",
            f"No CSVs to load in {csv_dir}/ - the loading job would be empty.",
            "",
            "  python -m src.graph.export --data data --out graph_csv",
            "  (or --data data_fixture --out graph_csv_fixture for the synthetic set)",
            "",
        ]))
    return "\n".join(lines) + "\n"


def tag_for_file(csv_name: str) -> str:
    for tag, name, *_ in MAPPINGS:
        if name == csv_name:
            return tag
    for tag, name, *_ in CONDITIONAL:
        if name == csv_name:
            return tag
    return ""


def files_to_upload(csv_dir: str) -> list[tuple[str, str]]:
    """(tag, path) for every CSV the job reads, deduplicated by tag."""
    seen: dict[str, str] = {}
    for tag, name, *_ in MAPPINGS + [(t, n, k, g, c) for t, n, k, g, c, _ in CONDITIONAL]:
        path = os.path.join(csv_dir, name)
        if os.path.exists(path):
            seen.setdefault(tag, path)
    return sorted(seen.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="graph_csv")
    ap.add_argument("--graph", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from .deploy import _graph_name
    body = generate(a.csv, a.graph or _graph_name())
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"wrote {a.out}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
