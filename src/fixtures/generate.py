"""A synthetic stand-in for the 704 MB dataset, with the same schema and the same case ids.

    python -m src.fixtures.generate            # writes fixtures/ then data_fixture/

WHY THIS EXISTS. Three reasons, in order of how much they matter.

  1. The real dataset ships separately and is not always on the machine. Without a stand-in,
     nothing downstream of `features/build.py` can be run or tested at all - which is how two
     policy bugs survived in a repo with 66 passing tests. The fixture makes the whole chain
     runnable: build -> backend -> investigation -> answer files -> validator.
  2. A judge cloning the repo does not have the 675 MB file either. `python -m src.fixtures.generate`
     followed by the runner gives them a working end-to-end demonstration in about ten seconds.
  3. It lets us plant known ground truth. The real pack's answers are hidden, so there is no way
     to assert that a subscription dispute is cleared or that a card-testing burst is detected.
     Here every case is built to a shape we chose, and `EXPECTED` below records it, so the
     investigation can be tested against intent rather than eyeballed.

WHAT IT IS NOT. It is not a substitute for the real answer files, and the numbers it produces
mean nothing about the real pack. It is 3,000 rows against 590,742, and the distributions are
hand-made. Anything measured here is a check that the machinery works, never a finding.

DESIGN NOTE. The generator emits CSVs in the organizers' format and then hands them to the real
`features/build.py`. It does not write parquet itself. That way the fixture exercises the actual
card-key reconstruction, the actual as-of baselines, the actual ring filter and the actual R5
detector - a fixture that bypassed them would test a parallel universe.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_CSV = os.path.join(ROOT, "fixtures")
FIXTURE_DATA = os.path.join(ROOT, "data_fixture")

SEED = 20260921
START = pd.Timestamp("2016-07-01 08:00:00")
FLAG_WINDOW_START = pd.Timestamp("2016-11-12 00:00:00")

# The 20 fixture cases, each built to a shape. `expect` is what a correct investigation should
# conclude - not asserted by this module, but asserted by tests/test_end_to_end.py.
#
# shape        what is planted                                      expected verdict
# -----------  ---------------------------------------------------  ----------------
# top_amount   largest amount the card has ever seen, new region    fraud
# subscription monthly charge the cardholder disputes (R7)          legitimate
# testing      R5 burst of sub-$5 online auths then a big purchase  fraud (card_testing)
# ring         rare full device profile shared across three cards   fraud (undocumented)
# benign       ordinary amount, known region/device/domain          legitimate
# ambiguous    mid-percentile amount, new device, nothing else      uncertain
CASES: list[tuple[str, str, str]] = [
    ("HHG-001", "benign", "risk_score"),
    ("HHG-002", "top_amount", "risk_score"),
    ("HHG-003", "subscription", "customer_report"),
    ("HHG-004", "top_amount", "customer_report"),
    ("HHG-005", "benign", "risk_score"),
    ("HHG-006", "top_amount", "customer_report"),
    ("HHG-007", "ambiguous", "risk_score"),
    ("HHG-008", "subscription", "customer_report"),
    ("HHG-009", "benign", "customer_report"),
    ("HHG-010", "top_amount", "risk_score"),
    ("HHG-011", "ambiguous", "customer_report"),
    ("HHG-012", "benign", "risk_score"),
    ("HHG-013", "benign", "risk_score"),
    ("HHG-014", "ring", "analyst_request"),
    ("HHG-015", "top_amount", "risk_score"),
    ("HHG-016", "ambiguous", "customer_report"),
    ("HHG-017", "ambiguous", "risk_score"),
    ("HHG-018", "subscription", "customer_report"),
    ("HHG-019", "testing", "risk_score"),
    ("HHG-020", "benign", "risk_score"),
]

def expected_verdict(shape: str, trigger: str) -> str | None:
    """What a correct investigation should conclude, given the shape AND how it arrived.

    The trigger matters as much as the transaction, which is the whole point of the exercise:

      - A dull transaction nobody disputed is legitimate. The same transaction reported by the
        cardholder is NOT, because nothing in the evidence explains why they are disputing it -
        a skimmed card is often used for something unremarkable. That case is `uncertain` and
        belongs with a human.
      - A subscription the cardholder disputes IS legitimate, because the card's own history
        explains the dispute. That is policy R7, and it is the difference between an agent that
        blocks on every complaint and one that reads the statement first.
      - An ambiguous transaction has NO single correct verdict, and returning None says so.
        Where it ends up depends on what the evidence loop asks and what comes back - one of
        these cases lands `legitimate` because a step-up challenge passed, which is the loop
        working rather than a wrong answer. The property worth asserting there is not the
        verdict but the restraint: an ambiguous case must never be decided without asking.
        `tests/test_end_to_end.py` asserts exactly that.
    """
    disputed = trigger == "customer_report"
    if shape in ("top_amount", "testing", "ring"):
        return "fraud"
    if shape == "subscription":
        # R7 does not prescribe a verdict, it prescribes a RESTRAINT: "do not block". A disputed
        # subscription can defensibly come out `legitimate` (the recurrence explains it) or
        # `uncertain` (it explains the charge but not why the cardholder is disputing it), and
        # which one depends on how far the other signals pull. So no verdict is asserted here -
        # `test_subscription_disputes_are_not_blocked` asserts the thing that actually matters.
        return None
    if shape == "ambiguous":
        return None
    return "uncertain" if disputed else "legitimate"     # benign


EXPECTED_PATTERN = {
    "testing": "card_testing",
    "ring": "undocumented",
}
"""Where the pattern is planted precisely enough to assert. The others depend on which of
several documented typologies fits best, which is a judgement the classifier should be free to
make - asserting it would be testing the fixture's taste, not the classifier's correctness."""

PRODUCTS_ONLINE = ["C", "R", "H", "S"]
DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com"]
REGIONS = [204.0, 264.0, 299.0, 330.0, 441.0, 469.0]

# A rare, fully specified profile - the only thing the ring filter will accept - and a generic
# one, which it must reject however many cards share it.
RARE_PROFILE = ("SM-G930V Build/NRD90M", "Android 7.0", "chrome 62.0 for android", "1440x2560")
GENERIC_PROFILE = ("Windows", "Windows 10", "chrome 63.0", "1920x1080")


def _stamp(ts: pd.Timestamp) -> str:
    """Second precision, like the real `ts` column.

    The generator does its arithmetic in fractional hours, which leaves microseconds on some
    rows and not others - and `pd.to_datetime` in the real build step infers a format from the
    first value and then fails on the rest. The organizers' file is uniform; the fixture must
    be too, or it tests a parser path the real data never takes.
    """
    return str(pd.Timestamp(ts).floor("s"))


class Builder:
    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)
        self.txns: list[dict] = []
        self.identity: list[dict] = []
        self.next_id = 3_000_001

    # --- primitives -------------------------------------------------------------------------

    def card_cols(self, n: int) -> dict:
        """Six card columns. The card key is the tuple, so these must be stable per card."""
        return {"card1": float(1000 + n), "card2": float(200 + n % 40),
                "card3": 150.0, "card4": "visa" if n % 3 else "mastercard",
                "card5": float(100 + n % 20), "card6": "debit" if n % 2 else "credit"}

    def txn(self, *, customer: str, cards: dict, ts: pd.Timestamp, amount: float,
            product: str, region: float | None, domain: str | None, risk: float,
            device: tuple | None = None, device_state: str = "Found",
            proxy: str | None = None) -> int:
        tid = self.next_id
        self.next_id += 1
        online = product != "W"
        self.txns.append({
            "TransactionID": tid,
            "TransactionDT": int((ts - START).total_seconds()) + 86400,
            "TransactionAmt": round(amount, 2),
            "ProductCD": product,
            **cards,
            "addr1": region, "addr2": 87.0, "dist1": np.nan,
            "P_emaildomain": domain, "R_emaildomain": np.nan,
            "customer_id": customer, "ts": _stamp(ts),
            "channel": "online" if online else "in_person",
            "risk_score": round(risk, 2),
        })
        if online and device is not None:
            info, os_, browser, screen = device
            self.identity.append({
                "TransactionID": tid, "DeviceType": "mobile" if "Android" in os_ else "desktop",
                "DeviceInfo": info, "id_15": device_state, "id_23": proxy,
                "id_30": os_, "id_31": browser, "id_33": screen,
            })
        return tid

    # --- per-case histories -----------------------------------------------------------------

    def routine_history(self, customer: str, cards: dict, *, n: int, region: float,
                        domain: str, product: str, device: tuple,
                        base_amount: float) -> pd.Timestamp:
        """A cardholder's ordinary life: steady amounts, one region, one device."""
        ts = START + pd.Timedelta(days=float(self.rng.integers(0, 10)))
        for _ in range(n):
            ts += pd.Timedelta(hours=float(self.rng.uniform(18, 96)))
            self.txn(customer=customer, cards=cards, ts=ts,
                     amount=float(self.rng.normal(base_amount, base_amount * 0.25)).__abs__() + 5,
                     product=product, region=region, domain=domain,
                     risk=float(self.rng.uniform(0.02, 0.35)), device=device)
        return ts

    def subscription_history(self, customer: str, cards: dict, *, amount: float, region: float,
                            domain: str, product: str, device: tuple, months: int,
                            first: pd.Timestamp) -> pd.Timestamp:
        """A monthly charge at a steady amount - the pattern R7 exists to protect."""
        ts = first
        for _ in range(months):
            self.txn(customer=customer, cards=cards, ts=ts,
                     amount=amount + float(self.rng.uniform(-0.05, 0.05)),
                     product=product, region=region, domain=domain,
                     risk=float(self.rng.uniform(0.05, 0.3)), device=device)
            ts += pd.Timedelta(days=30) + pd.Timedelta(hours=float(self.rng.integers(-12, 12)))
        return ts

    def testing_burst(self, customer: str, cards: dict, *, at: pd.Timestamp, region: float,
                      domain: str, device: tuple) -> None:
        """Three sub-$5 online authorisations inside an hour: the literal R5 reading."""
        ts = at
        for _ in range(4):
            self.txn(customer=customer, cards=cards, ts=ts,
                     amount=float(self.rng.uniform(0.8, 4.4)), product="C", region=region,
                     domain=domain, risk=float(self.rng.uniform(0.4, 0.8)), device=device,
                     device_state="New")
            ts += pd.Timedelta(minutes=float(self.rng.integers(5, 14)))


def build_fixture(seed: int = SEED) -> dict[str, pd.DataFrame]:
    b = Builder(seed)
    rng = b.rng
    pack_rows: list[dict] = []
    closed_rows: list[dict] = []
    case_no = 0

    # Filler customers with no exam case. Two jobs: the book is not made entirely of alerts, so
    # `similar_closed_cases` has a population to retrieve from, and - the reason there are 70 of
    # them rather than 6 - the GENERIC_PROFILE has to sit on enough cards to behave like a
    # genuinely common configuration. `rings.MAX_GLOBAL_CARDS_WEAK` is 60, so a fixture where
    # every card shared one profile made the filter treat the whole synthetic book as one ring.
    # In the real data that profile is on 842 customers; here it needs to clear 60.
    ring_cards: list[tuple[str, dict]] = []

    for idx, (case_id, shape, trigger) in enumerate(CASES):
        cust = f"C{idx + 1:05d}"
        cards = b.card_cols(idx)
        region = float(REGIONS[idx % len(REGIONS)])
        domain = DOMAINS[idx % len(DOMAINS)]
        product = PRODUCTS_ONLINE[idx % len(PRODUCTS_ONLINE)] if idx % 4 else "W"
        device = GENERIC_PROFILE
        base = float(rng.uniform(40, 130))
        n_hist = int(rng.integers(30, 90))

        if shape == "subscription":
            # history is the subscription plus ordinary activity around it
            last = b.routine_history(cust, cards, n=n_hist, region=region, domain=domain,
                                     product=product, device=device, base_amount=base)
            # Clearly outside the card's routine spending, which is N(base, 0.25*base): a
            # subscription drawn from the same range as everyday charges lands within the
            # detector's amount tolerance of them, which pollutes the interval series and makes
            # a perfectly regular subscription look irregular. The detector was right to say so;
            # the fixture was wrong to create it.
            sub_amount = round(base * 3.0 + 0.99, 2)
            first = FLAG_WINDOW_START - pd.Timedelta(days=30 * 4 + 2)
            nxt = b.subscription_history(cust, cards, amount=sub_amount, region=region,
                                         domain=domain, product=product, device=device,
                                         months=4, first=first)
            flag_ts = nxt
            flag_amount, flag_risk, flag_region = sub_amount, float(rng.uniform(0.2, 0.5)), region
            flag_device, flag_state, flag_proxy = device, "Found", None

        elif shape == "top_amount":
            last = b.routine_history(cust, cards, n=n_hist, region=region, domain=domain,
                                     product=product, device=device, base_amount=base)
            flag_ts = max(last, FLAG_WINDOW_START) + pd.Timedelta(days=float(rng.uniform(1, 30)))
            flag_amount = base * float(rng.uniform(8, 14))
            flag_risk = float(rng.uniform(0.2, 0.95))
            flag_region = float(rng.choice([r for r in REGIONS if r != region]))  # new region
            flag_device, flag_state, flag_proxy = device, "New", None

        elif shape == "testing":
            last = b.routine_history(cust, cards, n=n_hist, region=region, domain=domain,
                                     product=product, device=device, base_amount=base)
            burst_at = max(last, FLAG_WINDOW_START) + pd.Timedelta(days=float(rng.uniform(1, 20)))
            b.testing_burst(cust, cards, at=burst_at, region=region, domain=domain,
                            device=RARE_PROFILE)
            flag_ts = burst_at + pd.Timedelta(hours=2)
            flag_amount = base * 6
            flag_risk = float(rng.uniform(0.6, 0.95))
            flag_region, flag_device = region, RARE_PROFILE
            flag_state, flag_proxy = "New", None

        elif shape == "ring":
            last = b.routine_history(cust, cards, n=n_hist, region=region, domain=domain,
                                     product=product, device=device, base_amount=base)
            flag_ts = max(last, FLAG_WINDOW_START) + pd.Timedelta(days=float(rng.uniform(1, 20)))
            flag_amount = base * 3
            flag_risk = float(rng.uniform(0.3, 0.7))
            flag_region, flag_device = region, RARE_PROFILE
            flag_state, flag_proxy = "New", "ANONYMOUS"
            ring_cards.append((cust, cards))

        elif shape == "ambiguous":
            last = b.routine_history(cust, cards, n=n_hist, region=region, domain=domain,
                                     product=product, device=device, base_amount=base)
            flag_ts = max(last, FLAG_WINDOW_START) + pd.Timedelta(days=float(rng.uniform(1, 30)))
            flag_amount = base * float(rng.uniform(1.2, 2.0))
            flag_risk = float(rng.uniform(0.3, 0.8))
            flag_region, flag_device = region, GENERIC_PROFILE
            flag_state, flag_proxy = "New", None

        else:  # benign
            last = b.routine_history(cust, cards, n=n_hist, region=region, domain=domain,
                                     product=product, device=device, base_amount=base)
            flag_ts = max(last, FLAG_WINDOW_START) + pd.Timedelta(days=float(rng.uniform(1, 30)))
            flag_amount = base * float(rng.uniform(0.5, 0.9))
            flag_risk = float(rng.uniform(0.4, 0.9))     # a false alarm: high score, dull txn
            flag_region, flag_device = region, device
            flag_state, flag_proxy = "Found", None

        flag_product = product if shape != "testing" else "C"
        flagged = b.txn(customer=cust, cards=cards, ts=flag_ts, amount=flag_amount,
                        product=flag_product,
                        region=flag_region if flag_product != "C" else None,
                        domain=domain, risk=flag_risk,
                        device=flag_device, device_state=flag_state, proxy=flag_proxy)

        pack_rows.append({
            "case_id": case_id, "customer_id": cust, "card_id": f"{cust}-K1",
            "flagged_txn_id": flagged, "trigger_type": trigger,
            "risk_score": round(flag_risk, 2) if trigger == "risk_score" else np.nan,
            "opened_at": _stamp(flag_ts + pd.Timedelta(hours=1)),
            "narrative": {
                "customer_report": "Cardholder called: does not recognise this charge.",
                "risk_score": "Model score above alerting threshold.",
                "analyst_request": "Analyst asked for a review of connected activity.",
            }[trigger],
        })

        # Prior closed cases for two thirds of customers, mirroring the real pack's 16/20.
        if idx % 3 != 2:
            for _ in range(int(rng.integers(1, 4))):
                case_no += 1
                mine = [t for t in b.txns if t["customer_id"] == cust
                        and pd.Timestamp(t["ts"]) < FLAG_WINDOW_START - pd.Timedelta(days=14)]
                if not mine:
                    continue
                pick = mine[int(rng.integers(0, len(mine)))]
                fraud = bool(rng.random() < 0.84)          # the real history is 83.8% fraud
                opened = pd.Timestamp(pick["ts"]) + pd.Timedelta(hours=2)
                closed_rows.append({
                    "case_id": f"CC-{case_no:04d}", "customer_id": cust,
                    "card_id": f"{cust}-K1",
                    "opened_at": _stamp(opened),
                    "closed_at": _stamp(opened + pd.Timedelta(days=2)),
                    "outcome": "confirmed_fraud" if fraud else "cleared",
                    "pattern": (str(rng.choice(["card_not_present_fraud", "account_takeover",
                                                "card_not_present_new_device",
                                                "out_of_region_use"])) if fraud else "none"),
                    "exposure_usd": round(pick["TransactionAmt"], 2) if fraud else 0.0,
                    "n_txns": 1 if fraud else 0,
                    "txn_ids": str(pick["TransactionID"]),
                    "first_fraud_txn_id": pick["TransactionID"] if fraud else np.nan,
                    "connected_card_ids": np.nan,
                    "actions_taken": ("CREATE_CASE|BLOCK_CARD" if fraud
                                      else "VERIFY_WITH_CUSTOMER|CLOSE_NO_FRAUD"),
                    "report_filed": "Yes" if fraud and rng.random() < 0.071 else "No",
                    "analyst_notes": (
                        f"Cardholder {cust} disputed a ${pick['TransactionAmt']:.2f} "
                        f"{pick['channel']} transaction. "
                        + ("Confirmed fraudulent; card blocked and reissued."
                           if fraud else
                           "Cardholder recognised the charge on review; closed with no fraud.")),
                })

    # FILLER CUSTOMERS, generated LAST and from their own stream, deliberately. The builder shares one generator, so changing the
    # filler count shifted every draw after it and silently changed the shape of all 20 cases -
    # adding filler customers made a `benign` case come out `uncertain`. A fixture whose cases
    # depend on how much unrelated padding surrounds them is not a fixture. Filler now draws
    # from a separate stream, so the cases are stable under any filler count.
    filler_rng = np.random.default_rng(SEED + 1)
    for k in range(70):
        cust = f"C9{k:04d}"
        cards = b.card_cols(900 + k)
        saved, b.rng = b.rng, filler_rng
        b.routine_history(cust, cards, n=int(filler_rng.integers(8, 25)),
                          region=float(filler_rng.choice(REGIONS)),
                          domain=str(filler_rng.choice(DOMAINS)),
                          # ONLINE, not "W". In-person transactions carry no identity record, so
                          # filler generated as card-present contributed nothing to the device
                          # profiles - leaving GENERIC_PROFILE on only the 20 case cards, where
                          # the ring filter read it as a real shared origin and every case came
                          # back as a coordinated ring. The generic profile has to be genuinely
                          # generic for the filter to be tested at all.
                          product="C",
                          device=GENERIC_PROFILE,
                          base_amount=float(filler_rng.uniform(30, 120)))
        b.rng = saved

    # The ring: give the rare profile to two more cards inside a two-week window, so the filter
    # has something real to find - and keep the generic profile on many cards, so it has
    # something real to reject.
    if ring_cards:
        anchor_cust, _ = ring_cards[0]
        anchor_ts = max(pd.Timestamp(t["ts"]) for t in b.txns
                        if t["customer_id"] == anchor_cust)
        for k in range(2):
            cust = f"C801{k:02d}"
            cards = b.card_cols(801 + k)
            b.routine_history(cust, cards, n=12, region=float(rng.choice(REGIONS)),
                              domain=str(rng.choice(DOMAINS)), product="C",
                              device=GENERIC_PROFILE, base_amount=60.0)
            b.txn(customer=cust, cards=cards,
                  ts=anchor_ts - pd.Timedelta(days=float(rng.uniform(1, 8))),
                  amount=float(rng.uniform(80, 400)), product="C", region=None,
                  domain=str(rng.choice(DOMAINS)), risk=float(rng.uniform(0.4, 0.9)),
                  device=RARE_PROFILE, device_state="New", proxy="ANONYMOUS")

    txns = pd.DataFrame(b.txns)
    identity = pd.DataFrame(b.identity)
    pack = pd.DataFrame(pack_rows)
    closed = pd.DataFrame(closed_rows)
    return {"transactions": txns, "identity": identity, "case_pack": pack,
            "closed_cases_history": closed}


def write(frames: dict[str, pd.DataFrame], out_dir: str = FIXTURE_CSV) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    note = os.path.join(out_dir, "README.md")
    with open(note, "w", encoding="utf-8") as fh:
        fh.write(
            "# fixtures/ - SYNTHETIC DATA\n\n"
            "Generated by `python -m src.fixtures.generate`. Same schema and same case ids as "
            "the organizers' dataset, ~3,000 rows instead of 590,742.\n\n"
            "**Nothing measured here is a finding about the real data.** It exists so the whole "
            "chain - build, backend, investigation, answer files, validator - can be run and "
            "tested without the 704 MB download, and so the investigation can be asserted "
            "against planted ground truth instead of eyeballed.\n\n"
            "Expected verdicts per planted shape are in `src/fixtures/generate.py` "
            "(`EXPECTED_VERDICT`), and asserted by `tests/test_end_to_end.py`.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", default=FIXTURE_CSV)
    ap.add_argument("--out-data", default=FIXTURE_DATA)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--csv-only", action="store_true",
                    help="write the CSVs but do not run features/build.py over them")
    a = ap.parse_args()

    frames = build_fixture(a.seed)
    write(frames, a.out_csv)
    print(f"fixtures -> {a.out_csv}")
    for name, df in frames.items():
        print(f"  {name:<22} {len(df):>6,} rows")

    if a.csv_only:
        return 0

    # Hand the CSVs to the REAL pipeline, so the fixture exercises production code.
    from ..features.build import check_gate, run
    result = run(a.out_csv, a.out_data)
    print(f"\nbuild -> {a.out_data}")
    for k, v in result["counts"].items():
        print(f"  {k:<15} {v:>7,}")
    print(f"  provenance      {result['provenance']}")
    print(f"  card_testing    {result['card_testing_hits']} card(s) with a qualifying burst")
    print(f"  specific device profiles {result['specific_profiles']}")
    check_gate(result["counts"], expected=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
