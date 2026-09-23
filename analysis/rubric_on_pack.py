"""Run the whole deterministic assessment layer over all 20 exam cases, from a committed CSV.

    python analysis/rubric_on_pack.py

Rubric probability, pattern classification, policy recommendation and the value-of-information
choice of next question - for every case in the pack, with no dataset, no credentials and no
graph. `analysis/exam_triage.csv` is in git, so this runs on a fresh clone.

It is the acceptance check for the rubric (TRANSFER.md section 7, step 1), and it answers three
questions that a unit test cannot:

  1. Is the verdict split sane? A 20-0 split in either direction is a bug, not a finding: roughly
     half the pack is meant to be legitimate.
  2. Does the ordering contradict the cheap features? If the rubric ranks a 15th-percentile $36
     transaction above a 97th-percentile $482 one, something is wrong upstream.
  3. How many cases are still mid-band? Those are the cases where the evidence loop and the
     value-of-information selector earn their keep, so the number should not be zero.

What it deliberately does NOT model: ring signals, card-testing sequences, and anything else that
needs the graph. Those only ever push probabilities up, so the numbers here are a floor.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.context import build_policy_context  # noqa: E402
from src.agent.voi import evaluate_options, posture_label  # noqa: E402
from src.policy.rules import evaluate  # noqa: E402
from src.scoring.patterns import classify  # noqa: E402
from src.scoring.rubric import BAND_HIGH, BAND_LOW, RubricInput, score  # noqa: E402

TRIAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exam_triage.csv")


def _bool(v: str) -> bool:
    return str(v).strip().lower() == "true"


def _float(v: str) -> float | None:
    v = str(v).strip()
    return float(v) if v else None


def _proxy(v: str) -> str | None:
    """The triage table records `IP_PROXY:ANONYMOUS`; the rubric wants the bare state."""
    v = str(v).strip()
    return v.split(":")[-1] if v else None


def to_input(row: dict) -> RubricInput:
    """Map a triage row onto the rubric.

    The consistency columns (`known_region`, `known_email_domain`, `timing_typical`, ...) were
    added to `profile_dataset.py` after `exam_triage.csv` was last generated, so they are read
    with `.get()` and default to "not observed". Re-run
    `python analysis/profile_dataset.py` once the dataset is in place and they populate - which
    is what makes a `legitimate` verdict reachable without asking the cardholder.
    """
    return RubricInput(
        amt_pctile_prior=_float(row["amt_pctile"]),
        n_prior_txns=int(row["hist_txns"]),
        amount_usd=float(row["amount"]),
        channel=row["channel"].strip(),
        product_cd=row["product"].strip(),
        exposure_usd=float(row["amount"]),
        new_region=_bool(row["new_region"]),
        known_region=_bool(row.get("known_region", "")),
        new_product=_bool(row["new_product"]),
        known_product=_bool(row.get("known_product", "")),
        new_email_domain=_bool(row.get("new_email_domain", "")),
        known_email_domain=_bool(row.get("known_email_domain", "")),
        new_time_of_day=_bool(row.get("new_time_of_day", "")),
        timing_typical=_bool(row.get("timing_typical", "")),
        prior_txns_1h=int(row.get("prior_1h") or 0),
        prior_txns_24h=int(row.get("prior_24h") or 0),
        device_state=(row["id_15"].strip() or None),
        proxy=_proxy(row["proxy"]),
        risk_score=_float(row["risk_score"]),
        trigger_type=row["trigger"].strip(),
        prior_confirmed_fraud=int(row["prior_fraud"]),
        prior_cleared=int(row["prior_cleared"]),
    )


def _has_consistency_columns(rows: list[dict]) -> bool:
    return bool(rows) and "known_region" in rows[0]


def main() -> int:
    with open(TRIAGE, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print(f"{'case':<9} {'amt':>9} {'%ile':>5} {'risk':>5} {'p':>5} {'sig':>3} "
          f"{'verdict':<10} {'pattern':<28} {'posture':<14} next question")
    print("-" * 128)

    results = []
    for row in rows:
        i = to_input(row)
        s = score(i)
        exposure = float(row["amount"])
        pattern = classify(i, s)
        build = lambda ri, rs: build_policy_context(ri, rs, exposure_usd=exposure)  # noqa: E731
        decision = evaluate(build(i, s))
        selection = evaluate_options(i, build)
        results.append((row["case_id"], s, pattern, decision, selection))

        ask = selection.best.request_type if selection.best else "-- stop, nothing to gain"
        posture = posture_label(decision)
        print(f"{row['case_id']:<9} {exposure:>9,.2f} {float(row['amt_pctile']):>5.2f} "
              f"{float(row['risk_score']):>5.2f} {s.probability:>5.2f} "
              f"{s.n_independent_signals:>3} {s.verdict:<10} {pattern.pattern:<28} "
              f"{posture[:14]:<14} {ask}")

    print("\nRecommended actions per case (route in brackets)")
    for cid, _, _, decision, _ in results:
        acts = ", ".join(f"{r.action.value}[{r.route.value}]" for r in decision.recommendations)
        print(f"  {cid:<9} {acts}")

    print("\nDrivers")
    for cid, s, pattern, _, _ in results:
        drivers = ", ".join(f"{c.signal}{c.log_odds:+.1f}" for c in s.drivers)
        if not drivers:
            sub = [c for c in s.contributions if c.log_odds]
            drivers = ("~" + ", ~".join(f"{c.signal}{c.log_odds:+.1f}" for c in sub)
                       if sub else "no signal in the cheap features")
        print(f"  {cid:<9} {'[clamped] ' if s.clamped else ''}{drivers}")

    results = [(cid, s) for cid, s, _, _, _ in results]

    verdicts = [s.verdict for _, s in results]
    n = len(results)
    mid = [c for c, s in results if BAND_LOW < s.probability < BAND_HIGH]
    print("\nDistribution")
    for v in ("fraud", "uncertain", "legitimate"):
        print(f"  {v:<11} {verdicts.count(v):>2}/{n}")
    print(f"  mid-band    {len(mid):>2}/{n}  ({', '.join(mid) if mid else 'none'})")

    print("\nAcceptance checks (TRANSFER.md section 7, step 1)")
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'OK  ' if passed else 'FAIL'}] {name}{('  - ' + detail) if detail else ''}")

    check("verdict split is not unanimous",
          len(set(verdicts)) > 1, f"{dict((v, verdicts.count(v)) for v in set(verdicts))}")

    if _has_consistency_columns(rows):
        check("at least one case can be cleared without asking the cardholder",
              "legitimate" in verdicts,
              "a legitimate verdict needs two independent families, and profile consistency "
              "is the only one available before any evidence request")
    else:
        print("  [n/a ] legitimate verdicts not reachable from this CSV - it predates the "
              "consistency\n         columns, so `profile_fit` cannot fire. Re-run "
              "analysis/profile_dataset.py.")
    check("at least one case is still mid-band", bool(mid), f"{len(mid)} case(s)")
    check("no case leaves the band on one signal",
          all(BAND_LOW <= s.probability <= BAND_HIGH or s.n_independent_signals >= 2
              for _, s in results))

    # ordering sanity: the six top-percentile cases must all outrank the five bottom ones
    by_id = dict(results)
    top = ["HHG-002", "HHG-004", "HHG-006", "HHG-010", "HHG-011", "HHG-015"]
    bottom = ["HHG-005", "HHG-012", "HHG-013", "HHG-017", "HHG-018"]
    worst_top = min(by_id[c].probability for c in top)
    best_bottom = max(by_id[c].probability for c in bottom)
    check("top-percentile cases all outrank the low-percentile ones",
          worst_top > best_bottom,
          f"lowest of the six high-amount cases {worst_top:.2f} vs highest of the five "
          f"low-amount cases {best_bottom:.2f}")

    # the finding that motivated the rubric: risk_score must not drive the ranking
    check("HHG-006 (risk 0.25, 97th pctile) outranks HHG-013 (risk 0.76, 15th pctile)",
          by_id["HHG-006"].probability > by_id["HHG-013"].probability,
          f"{by_id['HHG-006'].probability:.2f} vs {by_id['HHG-013'].probability:.2f}")

    print("\nNote: ring signals and card-testing sequences are not modelled here (they need the"
          "\ngraph), and both only push probabilities up - so these numbers are a floor.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
