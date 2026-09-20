"""Validator for the 20 answer files - run before every submission.

    python -m src.io.validate cases/

"Missing fields score zero for that part", and several of the brief's requirements are
cross-field agreements a human proofread will not catch (sar.file must agree with FILE_REPORT;
exposure must equal the sum of the cited transactions; routes must match the policy table at
this case's exposure). This module checks all of them mechanically.

ERRORs are things that will lose points. WARNings are things that are probably wrong but might
be a defensible judgement call - a human should look.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import pandas as pd

from ..features.cards import Provenance
from ..policy.actions import Action, Route, route_for
from .answer import EVIDENCE_SOURCES, PATTERNS, REQUEST_TYPES, STATUSES, VERDICTS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

TOP_LEVEL = ["case_id", "case", "evidence_requests", "next_best_actions", "sar",
             "stop_reason", "tool_calls", "tokens", "latency_s"]
CASE_FIELDS = ["status", "verdict", "fraud_probability", "pattern", "pattern_description",
               "affected_txn_ids", "first_suspicious_txn_id", "connected_card_ids",
               "connected_device_profiles", "exposure_usd", "evidence", "similar_prior_cases",
               "summary", "written_to_graph", "graph_case_id"]
SAR_FIELDS = ["file", "reason", "narrative", "subjects", "total_amount_usd", "activity_dates"]


@dataclass
class Finding:
    case_id: str
    level: str      # ERROR | WARN
    rule: str
    message: str

    def __str__(self) -> str:
        return f"  [{self.level:<5}] {self.case_id} {self.rule}: {self.message}"


class Dataset:
    """The ID universe an answer file is allowed to reference."""

    def __init__(self, data_dir: str = DATA):
        t = pd.read_parquet(os.path.join(data_dir, "txns.parquet"),
                            columns=["TransactionID", "TransactionAmt", "ts", "customer_id",
                                     "card_key", "card_id"])
        self.txn_ids = set(t.TransactionID.astype(str))
        self.amounts = dict(zip(t.TransactionID.astype(str), t.TransactionAmt))
        self.timestamps = dict(zip(t.TransactionID.astype(str), t.ts.astype(str)))
        self.customer_ids = set(t.customer_id)

        cards = pd.read_parquet(os.path.join(data_dir, "cards.parquet"))
        self.citable_card_ids = set(cards.loc[cards.provenance != Provenance.UNKNOWN.value,
                                              "card_id"].dropna())
        self.cards_per_customer = cards.groupby("customer_id").size().to_dict()

        closed = pd.read_parquet(os.path.join(data_dir, "closed_cases.parquet"),
                                 columns=["case_id"])
        self.closed_case_ids = set(closed.case_id)

        pack = pd.read_parquet(os.path.join(data_dir, "case_pack.parquet"))
        self.pack = pack.set_index("case_id")
        self.expected_case_ids = set(pack.case_id)

        profiles = pd.read_parquet(os.path.join(data_dir, "device_profiles.parquet"),
                                   columns=["device_profile"])
        self.device_profiles = set(profiles.device_profile)


def _missing(obj: dict, fields: list[str]) -> list[str]:
    return [f for f in fields if f not in obj]


def validate_answer(a: dict, ds: Dataset) -> list[Finding]:
    cid = a.get("case_id", "<no case_id>")
    out: list[Finding] = []

    def err(rule, msg):
        out.append(Finding(cid, "ERROR", rule, msg))

    def warn(rule, msg):
        out.append(Finding(cid, "WARN", rule, msg))

    # 1 - structure
    for f in _missing(a, TOP_LEVEL):
        err("I1-structure", f"missing top-level field `{f}`")
    case, sar = a.get("case", {}), a.get("sar", {})
    nba = a.get("next_best_actions", {})
    for f in _missing(case, CASE_FIELDS):
        err("I1-structure", f"missing case.{f}")
    for f in _missing(sar, SAR_FIELDS):
        err("I1-structure", f"missing sar.{f}")
    for f in _missing(nba, ["initial", "final", "what_changed"]):
        err("I1-structure", f"missing next_best_actions.{f}")
    if not case or not nba:
        return out  # nothing further is checkable

    if cid not in ds.expected_case_ids:
        err("I1-structure", f"case_id is not in case_pack.csv")
    if case.get("status") not in STATUSES:
        err("I1-structure", f"status `{case.get('status')}` not in {sorted(STATUSES)}")
    if case.get("verdict") not in VERDICTS:
        err("I1-structure", f"verdict `{case.get('verdict')}` not in {sorted(VERDICTS)}")
    if case.get("pattern") not in PATTERNS:
        err("I1-structure", f"pattern `{case.get('pattern')}` not in {sorted(PATTERNS)}")
    p = case.get("fraud_probability")
    if not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
        err("I1-structure", f"fraud_probability `{p}` outside 0-1")

    verdict = case.get("verdict")
    affected = [str(t) for t in case.get("affected_txn_ids", [])]
    exposure = float(case.get("exposure_usd") or 0)
    final = nba.get("final", []) or []
    initial = nba.get("initial", []) or []
    final_names = [r.get("action") for r in final]

    # 2 - sar.file must agree with FILE_REPORT in the FINAL actions
    files = bool(sar.get("file"))
    if files != (Action.FILE_REPORT.value in final_names):
        err("I2-sar-agreement",
            f"sar.file={files} but FILE_REPORT {'is' if not files else 'is not'} in final actions")

    # 3 - a legitimate verdict carries no episode
    if verdict == "legitimate":
        if affected:
            err("I3-legitimate", f"verdict legitimate but {len(affected)} affected_txn_ids")
        if exposure:
            err("I3-legitimate", f"verdict legitimate but exposure ${exposure}")
        if files:
            err("I3-legitimate", "verdict legitimate but sar.file is true")

    # 4 - exposure must equal the sum of the cited transactions
    known = [t for t in affected if t in ds.amounts]
    if known:
        recomputed = round(sum(abs(ds.amounts[t]) for t in known), 2)
        if abs(recomputed - exposure) > 0.01:
            err("I4-exposure",
                f"exposure_usd {exposure} != sum of affected_txn_ids {recomputed}")

    # 5 - every id must exist, and cards must be citable
    for t in affected:
        if t not in ds.txn_ids:
            err("I5-ids", f"affected txn `{t}` is not in the dataset")
    first = str(case.get("first_suspicious_txn_id") or "")
    if first and first not in ds.txn_ids:
        err("I5-ids", f"first_suspicious_txn_id `{first}` is not in the dataset")
    if first and affected and first not in affected:
        warn("I5-ids", f"first_suspicious_txn_id `{first}` is not among affected_txn_ids")
    for c in case.get("connected_card_ids", []):
        if c not in ds.citable_card_ids:
            err("I5-ids", f"connected card `{c}` is not a resolvable card_id "
                          "(unknown K-suffix cards must not be cited)")
    for pc in case.get("similar_prior_cases", []):
        if pc not in ds.closed_case_ids:
            err("I5-ids", f"similar_prior_case `{pc}` is not in closed_cases_history.csv")
    for dp in case.get("connected_device_profiles", []):
        if dp not in ds.device_profiles:
            warn("I5-ids", f"device profile `{dp[:50]}...` not found verbatim in the data")
    for e in case.get("evidence", []):
        if e.get("source") not in EVIDENCE_SOURCES:
            err("I1-structure", f"evidence source `{e.get('source')}` invalid")
        for eid in e.get("entity_ids", []):
            s = str(eid)
            if not (s in ds.txn_ids or s in ds.citable_card_ids or s in ds.customer_ids
                    or s in ds.closed_case_ids or s in ds.device_profiles):
                warn("I5-ids", f"evidence entity_id `{s}` matches no known entity")

    # 6 - undocumented patterns must be described
    if case.get("pattern") == "undocumented" and not (case.get("pattern_description") or "").strip():
        err("I6-undocumented", "pattern is undocumented but pattern_description is empty")
    if case.get("pattern") != "undocumented" and (case.get("pattern_description") or "").strip():
        warn("I6-undocumented", "pattern_description set on a documented pattern (brief says \"\")")

    # 7 - routes must match the policy table at this exposure
    for label, actions in (("initial", initial), ("final", final)):
        for r in actions:
            act, got = r.get("action"), r.get("route")
            if act not in {a.value for a in Action}:
                err("I7-routes", f"{label}: `{act}` is not a policy action")
                continue
            want = route_for(act, exposure).value
            if got != want:
                err("I7-routes", f"{label}: {act} routed `{got}`, policy says `{want}` "
                                 f"at exposure ${exposure:,.2f}")
            if not (r.get("reason") or "").strip():
                err("I7-routes", f"{label}: {act} has no reason (policy section 7)")

    # 8 - no evidence requested means nothing changed
    reqs = a.get("evidence_requests", []) or []
    for r in reqs:
        if r.get("type") not in REQUEST_TYPES:
            err("I1-structure", f"evidence_request type `{r.get('type')}` invalid")
        if not (r.get("assumed_response") or "").strip():
            err("I8-evidence", "evidence_request has no assumed_response (brief requires it)")
    if not reqs:
        if initial != final:
            err("I8-evidence", "no evidence requested but initial != final")
        if (nba.get("what_changed") or "").strip().lower() != "nothing":
            err("I8-evidence", 'no evidence requested but what_changed != "nothing"')
    else:
        if initial == final and (nba.get("what_changed") or "").strip().lower() == "nothing":
            warn("I8-evidence", "evidence was requested but nothing changed - is that right?")

    # 9 / 10 - SAR internal consistency
    if not files:
        if (sar.get("narrative") or "").strip():
            err("I9-sar-empty", "sar.file is false but narrative is non-empty")
        if sar.get("subjects") or sar.get("activity_dates") or float(sar.get("total_amount_usd") or 0):
            err("I9-sar-empty", "sar.file is false but subjects/dates/amount are populated")
        if not (sar.get("reason") or "").strip():
            warn("I9-sar-empty", "sar.file is false with no reason given")
    else:
        narrative = (sar.get("narrative") or "").strip()
        if not narrative:
            err("I9-sar-empty", "sar.file is true but narrative is empty")
        else:
            sentences = narrative.count(". ") + narrative.count("? ") + narrative.endswith(".")
            if sentences < 6:
                warn("I10-sar-content",
                     f"narrative looks short (~{sentences} sentences; brief asks 6-12)")
        if abs(float(sar.get("total_amount_usd") or 0) - exposure) > 0.01:
            err("I10-sar-content",
                f"sar.total_amount_usd {sar.get('total_amount_usd')} != exposure {exposure}")
        dates = sar.get("activity_dates") or []
        if len(dates) != 2:
            err("I10-sar-content", f"activity_dates must be [first, last], got {dates}")
        elif known:
            want = sorted({ds.timestamps[t][:10] for t in known})
            if [want[0], want[-1]] != list(dates):
                err("I10-sar-content",
                    f"activity_dates {dates} != span of affected txns [{want[0]}, {want[-1]}]")
        if not sar.get("subjects"):
            err("I10-sar-content", "sar.file is true but subjects is empty")

    # 11 - R10
    if Action.BLOCK_ALL_CARDS.value in final_names:
        cust = ds.pack.customer_id.get(cid)
        n = ds.cards_per_customer.get(cust, 0)
        if n < 2:
            err("I11-r10", f"BLOCK_ALL_CARDS but customer {cust} holds {n} card(s) (R10)")

    # instrumentation
    for f in ("tool_calls", "tokens"):
        if not isinstance(a.get(f), int) or a.get(f) < 0:
            err("I1-structure", f"{f} must be a non-negative int, got {a.get(f)!r}")
    if a.get("tool_calls") == 0:
        warn("I1-structure", "tool_calls is 0 - was the run instrumented?")
    if not (a.get("stop_reason") or "").strip():
        err("I1-structure", "stop_reason is empty (required)")
    summary = (case.get("summary") or "").strip()
    if not summary:
        err("I1-structure", "case.summary is empty")
    elif summary.count(". ") > 8:
        warn("I1-structure", "summary is long; the brief asks for 2-6 sentences")

    return out


def validate_set(answers: dict[str, dict], ds: Dataset) -> list[Finding]:
    """Invariant 12: sanity of the distribution across all 20, not any single file."""
    out: list[Finding] = []
    if not answers:
        return out
    n = len(answers)
    missing = ds.expected_case_ids - set(answers)
    for c in sorted(missing):
        out.append(Finding(c, "ERROR", "I12-set", "no answer file for this case"))

    verdicts = [a.get("case", {}).get("verdict") for a in answers.values()]
    frauds = verdicts.count("fraud")
    if n >= 10 and frauds in (0, n):
        out.append(Finding("<set>", "ERROR", "I12-set",
                           f"every verdict is the same ({verdicts[0]}); roughly half the pack "
                           "is meant to be legitimate"))
    elif n >= 10 and frauds / n > 0.75:
        out.append(Finding("<set>", "WARN", "I12-set",
                           f"{frauds}/{n} verdicts are fraud - check calibration"))

    files = sum(1 for a in answers.values() if a.get("sar", {}).get("file"))
    if n >= 10 and files / n > 0.33:
        out.append(Finding("<set>", "WARN", "I12-set",
                           f"{files}/{n} cases file a report; closed-case history rate is 7.1%"))

    testing = sum(1 for a in answers.values()
                  if a.get("case", {}).get("pattern") == "card_testing")
    if testing:
        out.append(Finding("<set>", "WARN", "I12-set",
                           f"{testing} case(s) called card_testing; no exam case has a "
                           "qualifying R5 sequence at its flagged moment"))

    probs = [a.get("case", {}).get("fraud_probability", 0) for a in answers.values()]
    if n >= 10 and all(x > 0.8 or x < 0.2 for x in probs):
        out.append(Finding("<set>", "WARN", "I12-set",
                           "no case lands mid-band; `uncertain` is a valid verdict and some "
                           "cases are designed to be ambiguous"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="cases", help="directory of answer files")
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()

    ds = Dataset(a.data)
    paths = sorted(glob.glob(os.path.join(a.path, "*.json")))
    if not paths:
        print(f"no answer files in {a.path}/", file=sys.stderr)
        return 1

    answers, findings = {}, []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                obj = json.load(fh)
        except json.JSONDecodeError as e:
            findings.append(Finding(os.path.basename(p), "ERROR", "I1-structure", f"invalid JSON: {e}"))
            continue
        answers[obj.get("case_id", os.path.basename(p))] = obj
        findings.extend(validate_answer(obj, ds))
    findings.extend(validate_set(answers, ds))

    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]
    for f in findings:
        print(f)
    print(f"\n{len(answers)}/{len(ds.expected_case_ids)} cases | "
          f"{len(errors)} errors | {len(warns)} warnings")
    if errors:
        print("\nNOT SUBMITTABLE - fix the errors above.", file=sys.stderr)
        return 1
    print("\nAll invariants pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
