"""Write finished answer files into TigerGraph as Case vertices, and say so truthfully.

    python -m src.graph.publish --cases cases

WHY THIS IS A SEPARATE STEP. The brief requires the case to be written into the graph - it is
the case memory the next investigation retrieves - and `written_to_graph` in the answer file has
to reflect a real write rather than a constant. The investigation can do that itself when it runs
with `--backend tigergraph`, but the answers we ship are produced by the parquet backend, for a
measured reason:

the two backends disagree on shared-device detection at the moment, and the parquet one is right.
`export.py` precomputes `SHARES_DEVICE.span_days` over each card pair's ENTIRE history, so the
GSQL filter `span_days <= 14` rejects a profile that four cards used inside one fortnight if that
profile was also active months earlier. HHG-014's ring - 52 cards, 52 customers, two tight bursts
across August and November - is exactly that shape, and the graph backend returned `legitimate`
at probability 0.04 on the one case whose trigger explicitly asks about a shared device profile.
The parquet backend computes the span inside the window, which is what the calibrated filter in
`features/rings.py` actually specifies.

So: the parquet backend decides, and this module publishes the result to the graph. Both halves
of the requirement are met and neither is faked. The proper fix is a window-scoped GSQL traversal
over ON_DEVICE instead of the precomputed edge, which is a query change and a reinstall - written
up in PLAN.md rather than rushed the night before a deadline.

Every write is verified by reading the case back, and the answer file is only marked
`written_to_graph: true` if that read-back succeeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def publish(cases_dir: str, dry_run: bool = False) -> dict:
    from .tigergraph import from_env

    paths = sorted(p for p in glob.glob(os.path.join(cases_dir, "*.json"))
                   if not os.path.basename(p).startswith("_"))
    if not paths:
        raise SystemExit(f"no answer files in {cases_dir}/ - run src.agent.run first")

    diag_path = os.path.join(cases_dir, "_diagnostics.json")
    diagnostics = {}
    if os.path.exists(diag_path):
        with open(diag_path, encoding="utf-8") as fh:
            diagnostics = json.load(fh)

    be = None if dry_run else from_env()
    written, failed = [], []

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            answer = json.load(fh)
        case_id = answer["case_id"]
        case = answer["case"]
        d = diagnostics.get(case_id, {})
        card = d.get("card") or {}
        trigger = d.get("trigger") or {}

        record = {
            "case_id": case_id,
            "graph_case_id": f"CASE-{case_id}",
            "customer_id": (card.get("card_id") or "").split("-")[0] or "",
            "card_id": card.get("card_id") or "",
            "card_key": card.get("card_key") or "",
            "opened_at": trigger.get("opened_at", ""),
            "status": case["status"],
            "verdict": case["verdict"],
            "fraud_probability": case["fraud_probability"],
            "pattern": case["pattern"],
            "exposure_usd": case["exposure_usd"],
            "stop_reason": answer.get("stop_reason", ""),
            "summary": case["summary"],
            "actions": [a["action"] for a in answer["next_best_actions"]["final"]],
            "affected_txn_ids": case["affected_txn_ids"],
            "similar_prior_cases": case["similar_prior_cases"],
            "evidence": case["evidence"],
            "decisions": d.get("decisions", []),
        }

        if dry_run:
            print(f"  {case_id}: would write CASE-{case_id} "
                  f"({len(record['evidence'])} evidence items)")
            written.append(case_id)
            continue

        try:
            gid = be.write_case(record)
            # `written_to_graph` means the graph can find it, not that a POST returned 200
            back = be.read_case(gid)
            if back is None:
                raise RuntimeError("written but not readable")
            case["written_to_graph"] = True
            case["graph_case_id"] = gid
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(answer, fh, indent=2, ensure_ascii=False)
            written.append(case_id)
            print(f"  {case_id} -> {gid}  (read back: verdict={back.get('verdict')}, "
                  f"{len(back.get('evidence') or [])} evidence items)")
        except Exception as exc:                            # noqa: BLE001 - reported per case
            failed.append((case_id, f"{type(exc).__name__}: {exc}"))
            case["written_to_graph"] = False
            case["graph_case_id"] = ""
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(answer, fh, indent=2, ensure_ascii=False)
            print(f"  {case_id} FAILED: {exc}")

    return {"written": written, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="cases")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print(f"publishing {a.cases}/ to the graph")
    result = publish(a.cases, a.dry_run)
    print(f"\n{len(result['written'])} written, {len(result['failed'])} failed")
    for case_id, why in result["failed"]:
        print(f"  {case_id}: {why}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
