"""Does this charge match a recurring pattern the cardholder already has?

WHY THIS IS ON THE CRITICAL PATH. Policy R7 says that when a disputed charge matches the
cardholder's own recurring pattern, do not block - it is a subscription the cardholder forgot,
not fraud. R7 is also the ONLY brake on R2, which blocks the card whenever the cardholder
disputes a transaction. Eight of the 20 exam cases arrive as customer reports. Without a
detector, all eight block, including HHG-003, HHG-009 and HHG-018, whose amounts sit at the
28th, 30th and 18th percentile of their own card's history - the shape of a routine charge, not
of a theft.

THE HARD PART: THERE IS NO MERCHANT COLUMN. IEEE-CIS carries no merchant id, so "same merchant"
is not directly observable. What is observable is the signature a subscription leaves on a
statement:

  - a near-identical amount, repeatedly
  - at a regular interval
  - through the same product code, and where recorded the same counterparty email domain
  - with the flagged transaction falling where the next one was due

That is a weaker claim than "same merchant", and the detector says so in its own words rather
than overstating what it knows: the evidence it emits names the recurrence it found, not a
merchant it cannot see.

EVERY READ IS AS-OF. A subscription is established by what came BEFORE the flagged transaction.
Letting the detector see later charges would clear cases using their own future, and would
inflate the closed-case backtest exactly where it matters most.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

AMOUNT_TOLERANCE_PCT = 0.02
"""How much a recurring amount may drift and still count as the same charge. Subscriptions
re-bill at the same price; 2% absorbs rounding and small tax changes without swallowing a
different purchase that happens to cost about the same."""

AMOUNT_TOLERANCE_ABS = 0.50
"""Floor for the tolerance, so a $3.99 charge is not held to a 8-cent window."""

MIN_OCCURRENCES = 2
"""Prior charges needed before a pattern is established. Two priors plus the flagged
transaction is three occurrences - the minimum that can show an interval at all, let alone a
regular one."""

CADENCES = {"weekly": 7.0, "fortnightly": 14.0, "monthly": 30.44, "quarterly": 91.3,
            "annual": 365.25}
CADENCE_TOLERANCE = 0.25
"""A monthly subscription lands anywhere from day 28 to day 33, and a retry after a declined
card can push it further. 25% of the nominal interval is loose enough for that and tight enough
to reject a coincidence."""

INTERVAL_CV_MAX = 0.35
"""Maximum coefficient of variation across the observed intervals. Regularity is the whole
claim: three charges of the same amount at 3, 40 and 210 days apart are not a subscription."""


@dataclass
class Recurrence:
    """A recurring pattern found in the card's own prior history, or the absence of one."""
    matches: bool
    cadence: str = ""
    interval_days: float = 0.0
    n_prior: int = 0
    prior_txn_ids: tuple[str, ...] = ()
    amount_usd: float = 0.0
    same_domain: bool = False
    reason: str = ""

    def as_evidence_claim(self) -> str:
        """Phrasing for the case file. Claims recurrence, not a merchant we cannot see."""
        if not self.matches:
            return self.reason
        domain = (" through the same purchaser email domain" if self.same_domain
                  else " (no purchaser domain recorded to corroborate the counterparty)")
        return (f"The disputed amount of ${self.amount_usd:,.2f} has been charged to this card "
                f"{self.n_prior} time(s) before at a {self.cadence} cadence "
                f"(median {self.interval_days:.1f} days){domain}, and the flagged transaction "
                f"falls where the next one was due - the signature of an established recurring "
                f"charge rather than a one-off")


def _tolerance(amount: float) -> float:
    return max(abs(amount) * AMOUNT_TOLERANCE_PCT, AMOUNT_TOLERANCE_ABS)


def _classify_cadence(interval_days: float) -> str:
    for name, nominal in CADENCES.items():
        if abs(interval_days - nominal) <= nominal * CADENCE_TOLERANCE:
            return name
    return ""


def detect(prior: pd.DataFrame, *, amount: float, ts: pd.Timestamp,
           product_cd: str | None = None, email_domain: str | None = None,
           n_prior_txns: int | None = None) -> Recurrence:
    """Look for a recurring charge matching `amount` in this card's transactions before `ts`.

    `prior` must already be restricted to one card and to transactions strictly earlier than
    `ts` - the caller owns the as-of cut, because the caller is the one holding the graph query.

    `n_prior_txns` is how many transactions the card has before `ts` IN TOTAL, and it matters
    because the two backends hand this function different things. The parquet backend passes the
    card's whole history and lets the amount filter below do the work; the GSQL backend gets the
    amount filtering done in the query and passes only the survivors. Without being told the
    total, an empty `prior` is ambiguous - it could mean "this card has never been used" or "this
    card has 358 transactions and none of them are for this amount" - and the two produce very
    different evidence claims. The graph backend was emitting the first about cards for which the
    second was true.
    """
    total = n_prior_txns if n_prior_txns is not None else (0 if prior is None else len(prior))
    if total <= 0:
        return Recurrence(False, reason="The card has no prior history to establish a pattern in")

    if prior is None or prior.empty:
        prior = pd.DataFrame(columns=["TransactionID", "ts", "TransactionAmt", "ProductCD",
                                      "P_emaildomain"])
    else:
        prior = prior[prior.ts < ts]

    tol = _tolerance(amount)
    same_amount = prior[(prior.TransactionAmt - amount).abs() <= tol]
    if product_cd and "ProductCD" in same_amount.columns:
        same_amount = same_amount[same_amount.ProductCD == product_cd]

    # The purchaser domain is the only counterparty-ish field in the dataset, and it is null on
    # 94,480 rows. Use it to strengthen a match, never to reject one.
    same_domain = False
    if email_domain and "P_emaildomain" in same_amount.columns:
        with_domain = same_amount[same_amount.P_emaildomain == email_domain]
        if len(with_domain) >= MIN_OCCURRENCES:
            same_amount, same_domain = with_domain, True

    if len(same_amount) < MIN_OCCURRENCES:
        return Recurrence(
            False, n_prior=len(same_amount),
            reason=(f"Only {len(same_amount)} prior charge(s) on this card fall within "
                    f"${tol:,.2f} of ${amount:,.2f}, which is too few to establish a recurring "
                    "pattern"))

    # The pattern is established by the PRIOR charges alone, and the flagged transaction is then
    # tested against it. Folding the flagged charge into the interval series instead would let an
    # off-cycle charge break the regularity it is being measured against, and the detector would
    # report "irregular spacing" about a subscription that is in fact perfectly regular.
    stamps = sorted(same_amount.ts.tolist())
    intervals = [(b - a).total_seconds() / 86400 for a, b in zip(stamps, stamps[1:])]
    median = float(pd.Series(intervals).median())
    mean = float(pd.Series(intervals).mean())
    # With exactly two prior charges there is one interval and no spread to measure; the cadence
    # check and the on-cycle check below still have to pass, so this is not a free ride.
    cv = (float(pd.Series(intervals).std(ddof=0) / mean)
          if len(intervals) > 1 and mean > 0 else 0.0)

    if cv > INTERVAL_CV_MAX:
        return Recurrence(
            False, n_prior=len(same_amount), amount_usd=amount, interval_days=median,
            reason=(f"{len(same_amount)} prior charges of about ${amount:,.2f} exist on this "
                    f"card but their spacing is irregular (intervals "
                    f"{', '.join(f'{i:.0f}d' for i in intervals)}), so this is a repeated amount "
                    "rather than a recurring charge"))

    cadence = _classify_cadence(median)
    if not cadence:
        return Recurrence(
            False, n_prior=len(same_amount), amount_usd=amount, interval_days=median,
            reason=(f"{len(same_amount)} prior charges of about ${amount:,.2f} recur every "
                    f"{median:.1f} days, which matches no standard billing cadence"))

    # the flagged transaction must itself land where the next charge was due
    gap = (ts - stamps[-1]).total_seconds() / 86400
    if abs(gap - median) > median * CADENCE_TOLERANCE:
        return Recurrence(
            False, cadence=cadence, n_prior=len(same_amount), amount_usd=amount,
            interval_days=median,
            reason=(f"This card does carry a {cadence} charge of about ${amount:,.2f}, but the "
                    f"flagged transaction arrives {gap:.1f} days after the last one instead of "
                    f"the usual {median:.1f}, so it is off-cycle"))

    return Recurrence(
        True, cadence=cadence, interval_days=median, n_prior=len(same_amount),
        # sorted: the order the rows arrive in is an artefact of the backend (a pandas groupby
        # versus a GSQL ACCUM), it carries no meaning, and leaving it unsorted made the two
        # backends emit the same five transaction ids in different orders in `entity_ids`
        prior_txn_ids=tuple(sorted(str(t) for t in same_amount.TransactionID.tolist())),
        amount_usd=amount, same_domain=same_domain)
