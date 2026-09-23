"""`fraud_probability` from a transparent, auditable rubric.

WHY A RUBRIC AND NOT A MODEL. `src/scoring/diagnose.py` has the measurement: all 900 cleared
closed cases carry `risk_score` >= 0.81, so below that the history holds 4,038 fraud examples and
zero legitimate ones. A classifier fitted there learns `low score => fraud` and posts a fake AUC
of 0.991 - and 17 of the 20 exam cases live in exactly that region. So the probability comes from
weights grounded in measured base rates instead of fitted to a biased label set. The trade is
deliberate: we lose whatever signal a model might have found, and we gain a number whose drivers
can be cited as evidence in the answer file.

HOW IT WORKS. Additive in log-odds from a stated prior, then a logistic. Every contribution is
named, signed, and attributed to an evidence FAMILY. Two things fall out of that structure:

  1. `drivers` - the two or three signals the number actually rests on, which is what the
     Assess Uncertainty node logs and what the case summary quotes.
  2. Independence counting - the policy's stopping rule (section 6) needs "two independent pieces
     of evidence", and "independent" has to mean something. Here it means two different families
     each contributing at least MATERIAL log-odds. Two amount-based observations are one signal.

WEIGHTS ARE CALIBRATED TO RARITY, NOT TO OUTCOMES. A signal present on 43% of the book cannot
separate anything, however suspicious it sounds; a signal present on 3.8% can. That is why
`id_15 = New` moves the number by almost nothing and a rare shared device inside 14 days moves it
a lot. The specific measurements behind each weight are in the comment on each constant.

Nothing here touches pandas, the graph, or the LLM: primitives in, score out. The agent's job is
to fill `RubricInput` honestly from the graph; this module's job is to turn it into the same
number every time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------
# Families. A "signal" is independent of another only if it comes from a different family.
# --------------------------------------------------------------------------------------------
AMOUNT = "amount_behaviour"
REGION = "geography"
DEVICE = "device"
RING = "shared_origin"
SEQUENCE = "sequence_velocity"
CUSTOMER = "customer_response"
HISTORY = "prior_cases"
TESTING = "card_testing"
MODEL = "risk_model"
PRODUCT = "product_mix"
FIT = "profile_fit"
COUNTERPARTY = "counterparty"
TIMING = "timing"

BASE_PRIOR = 0.30
"""Prior that an alert reaching investigation is fraud, before any case-specific evidence.

Not the closed-case base rate (83.8%), which is a filtered population of alerts someone already
thought worth opening, and not 50% either. The brief says roughly half the exam pack is meant to
be legitimate, and only 2.9% of all transactions score above 0.7 - so an alert starts below even
and has to earn its way up. Starting at 0.30 also means a single signal cannot reach the
0.85 stop threshold on its own, which is the behaviour policy section 6 asks for.
"""

MATERIAL = 0.40
"""Minimum |log-odds| for a contribution to count as an independent signal.

Set just above the largest weight `risk_score` can contribute (0.35) and far above `id_15 = New`
(0.15). So the two signals the brief warns are not evidence on their own mechanically cannot be
the thing that lets the probability leave the uncertainty band. That is the point.
"""

BAND_LOW, BAND_HIGH = 0.15, 0.85
"""Policy section 6: the probability may not leave this band on fewer than two independent
signals. Enforced in `score()` by clamping, not by hoping the prompt says so."""

CONFIRMED_CAP = 0.10
"""Ceiling on the probability once the cardholder confirms they made the transaction.

Policy R3 is unconditional - customer confirms, close with no fraud - so the rubric must not be
able to out-vote it with anomaly signals. A $500 transaction at the 99th percentile in a new
region that the cardholder says they made is an unusual purchase, not fraud.

The exception is account takeover: if the account itself looks compromised, whoever answered may
not be the cardholder, and a confirmation is then worth very little. So the cap is withheld when
a shared-origin ring, a card-testing sequence, or a failed step-up is on the record.
"""

FRAUD_AT = 0.70
LEGITIMATE_AT = 0.20
"""Verdict thresholds. Deliberately asymmetric: calling something legitimate closes a case and
stops all protective action, so it needs more confidence than calling it suspicious, which only
ever leads to verification or monitoring under R1."""


@dataclass
class Contribution:
    signal: str
    family: str
    log_odds: float
    note: str

    @property
    def material(self) -> bool:
        return abs(self.log_odds) >= MATERIAL

    @property
    def direction(self) -> str:
        return "raises" if self.log_odds > 0 else "lowers"

    def as_evidence_claim(self) -> str:
        """Phrasing that can go straight into the case file's evidence list."""
        return f"{self.note} ({self.direction} fraud probability, {self.family})"


@dataclass
class RubricInput:
    """Everything the rubric reads. Filled from the graph, never from the model's imagination.

    Defaults are the "nothing observed" case, so a caller that knows only the amount percentile
    still gets a defensible number.
    """
    # what the transaction is
    channel: str = "online"                  # online | in_person (a restatement of ProductCD)
    product_cd: str = ""
    n_affected_txns: int = 1
    exposure_usd: float = 0.0

    # amount behaviour - the best cheap discriminator in the pack
    amt_pctile_prior: float | None = None
    n_prior_txns: int = 0
    amount_usd: float = 0.0

    # geography / product / counterparty / timing.
    # The `new_*` flags mean "this card has never used this value before"; the `known_*` flags
    # mean "it has". They are not each other's negation - a null `addr1` is neither, and 94.8%
    # of C-product transactions have no region at all. Both directions are recorded explicitly
    # so that "no region on file" can never be read as either novelty or consistency.
    new_region: bool = False
    known_region: bool = False
    new_product: bool = False
    known_product: bool = False
    new_email_domain: bool = False
    known_email_domain: bool = False
    timing_typical: bool = False
    new_time_of_day: bool = False

    # device (online only; in-person transactions carry no identity record, which is normal)
    device_state: str | None = None          # New | Found | Unknown | None
    proxy: str | None = None                 # ANONYMOUS | HIDDEN | TRANSPARENT | None
    device_profile_is_full: bool = False
    device_profile_global_cards: int | None = None

    # shared origin, AFTER the calibrated ring filter in features/rings.py
    ring_signal: bool = False
    ring_volume_artefact: bool = False
    ring_n_cards: int = 0
    ring_n_customers: int = 0
    ring_strength: str = "strong"   # strong | moderate, see features/rings.link_strength
    ring_profile_cards: int = 0     # how many cards the profile touches book-wide

    # sequence
    prior_txns_1h: int = 0
    prior_txns_24h: int = 0
    card_testing: bool = False
    card_testing_escalated: bool = False

    # the bank's own model
    risk_score: float | None = None

    # trigger and any evidence that came back
    trigger_type: str = "risk_score"         # risk_score | customer_report | analyst_request
    customer_response: str | None = None     # denied | confirmed | no_reply | None
    step_up_result: str | None = None        # passed | failed | None
    matches_recurring_pattern: bool = False

    # as-of case history for this customer
    prior_confirmed_fraud: int = 0
    prior_cleared: int = 0


@dataclass
class RubricScore:
    probability: float
    contributions: list[Contribution] = field(default_factory=list)
    clamped: bool = False
    prior: float = BASE_PRIOR
    override: str = ""
    """Set when a policy rule is dispositive and the weighted sum must not be able to out-vote
    it. Carries the citation, so the case file can say which rule decided."""
    override_verdict: str = ""

    @property
    def material_contributions(self) -> list[Contribution]:
        return [c for c in self.contributions if c.material]

    @property
    def families(self) -> list[str]:
        seen: list[str] = []
        for c in self.material_contributions:
            if c.family not in seen:
                seen.append(c.family)
        return seen

    @property
    def n_independent_signals(self) -> int:
        """Distinct evidence families carrying a material contribution. See MATERIAL."""
        return len(self.families)

    @property
    def conflicting(self) -> bool:
        """Material evidence pointing both ways.

        Policy R8 escalates when evidence conflicts, and this is how the agent knows. HHG-017 is
        the shape: the amount sits at the 21st percentile of the card's own history (exonerating)
        while the connection comes through a hidden proxy (incriminating). Averaging those into
        a mid-band number and calling it uncertainty loses the reason it is uncertain - the case
        needs a human, not another decimal place.
        """
        pos = any(c.log_odds >= MATERIAL for c in self.contributions)
        neg = any(c.log_odds <= -MATERIAL for c in self.contributions)
        return pos and neg

    @property
    def drivers(self) -> list[Contribution]:
        """The two or three signals the number rests on, largest magnitude first."""
        return sorted(self.material_contributions,
                      key=lambda c: abs(c.log_odds), reverse=True)[:3]

    @property
    def verdict(self) -> str:
        """fraud | legitimate | uncertain - and `uncertain` unless two families agree.

        The band clamp guards the probability; this guards the label. Without it a single
        top-percentile amount produces "fraud" at 0.78 on one piece of evidence, which is
        precisely the overconfidence policy section 6 exists to prevent. Returning `uncertain`
        here is not indecision - it is what sends the case into the evidence loop, where the
        graph or a policy-approved request supplies the second signal.
        """
        if self.override_verdict:
            return self.override_verdict
        if self.n_independent_signals < 2:
            return "uncertain"
        if self.probability >= FRAUD_AT:
            return "fraud"
        if self.probability <= LEGITIMATE_AT:
            return "legitimate"
        return "uncertain"

    @property
    def verdict_blocked_by_independence(self) -> bool:
        """True when the probability is decisive but rests on a single family.

        The agent surfaces this as its reason for asking rather than acting, and it is the
        condition the value-of-information selector exists to resolve.
        """
        return (self.n_independent_signals < 2
                and (self.probability >= FRAUD_AT or self.probability <= LEGITIMATE_AT))

    def explain(self) -> str:
        """One line per contribution, in the order they were applied. Goes in the decision log."""
        lines = [f"prior {self.prior:.2f} (logit {_logit(self.prior):+.2f})"]
        for c in self.contributions:
            mark = " " if c.material else "~"   # ~ = present but not independent evidence
            lines.append(f"{mark} {c.log_odds:+.2f}  {c.signal:<26} {c.note}")
        lines.append(f"= {self.probability:.2f}"
                     f"{'  [clamped to the uncertainty band]' if self.clamped else ''}"
                     f"  on {self.n_independent_signals} independent signal(s)")
        if self.override:
            lines.append(f"! override -> {self.override}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "fraud_probability": round(self.probability, 3),
            "prior": self.prior,
            "n_independent_signals": self.n_independent_signals,
            "families": self.families,
            "clamped_to_band": self.clamped,
            "override": self.override,
            "drivers": [{"signal": c.signal, "family": c.family,
                         "log_odds": round(c.log_odds, 3), "note": c.note}
                        for c in self.drivers],
            "all_contributions": [{"signal": c.signal, "family": c.family,
                                   "log_odds": round(c.log_odds, 3), "note": c.note,
                                   "material": c.material}
                                  for c in self.contributions],
        }


AMT_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.00, -1.60),   # the smallest amounts this card transacts
    (0.10, -1.60),
    (0.25, -1.20),
    (0.40,  0.00),
    (0.80,  0.00),   # flat through the middle: an ordinary amount says nothing either way
    (0.90,  0.60),
    (0.95,  1.20),
    (0.99,  1.60),
    (1.00,  1.60),   # the largest amount the card has ever authorised
)
"""Amount-percentile weight, interpolated rather than tiered.

The anchors are the measured tiers; interpolating between them removes the cliff. With hard
tiers, HHG-003 at the 28th percentile scored exactly zero while the 25th percentile scored
-1.20 - a 0.03 change in the input switching a material signal on and off. Flat between 0.40
and 0.80 because an ordinary amount for this card is genuinely uninformative, and no amount of
curve fitting should pretend otherwise.
"""


def _interp(x: float, anchors: tuple[tuple[float, float], ...]) -> float:
    """Piecewise-linear lookup, clamped at both ends."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1:
            span = x1 - x0
            return y0 if span == 0 else y0 + (y1 - y0) * (x - x0) / span
    return anchors[-1][1]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------------------------
# The weights. Each one carries the measurement that justifies it.
# --------------------------------------------------------------------------------------------

def _amount(i: RubricInput, add) -> None:
    """Where the amount sits in this card's OWN prior distribution.

    Finding (4) in PROJECT.md section 3: six of the 20 exam cases sit at or near the top of their
    own card's history, and the percentile ranks them better than `risk_score` does - the score
    puts HHG-013 (0.76, 15th percentile, $36) above HHG-006 (0.25, 97th percentile, $482). A
    percentile also means the same thing on a 36-transaction card and a 10,306-transaction card,
    which raw amounts do not.
    """
    p = i.amt_pctile_prior
    if p is None:
        add("amt_pctile_prior", AMOUNT, 0.0,
            "no prior transactions on this card to compare the amount against")
        return
    if i.n_prior_txns < 5:
        # a percentile over four transactions is noise dressed as a statistic
        add("amt_pctile_prior", AMOUNT, 0.0,
            f"amount percentile {p:.2f} withheld - only {i.n_prior_txns} prior transaction(s) "
            "on this card, too few for a meaningful distribution")
        return

    w = _interp(p, AMT_ANCHORS)
    if w >= 1.40:
        note = (f"${i.amount_usd:,.2f} is the largest or near-largest amount this card has ever "
                f"authorised - {p:.0%} of its {i.n_prior_txns} prior transactions were smaller")
    elif w > 0:
        note = f"${i.amount_usd:,.2f} sits at the {p:.0%} percentile of this card's own history"
    elif w <= -1.40:
        note = (f"${i.amount_usd:,.2f} is among the smallest amounts this card transacts "
                f"({p:.0%} percentile) - the classic false-alarm shape")
    elif w < 0:
        note = f"${i.amount_usd:,.2f} is low for this card ({p:.0%} percentile)"
    else:
        note = f"${i.amount_usd:,.2f} is unremarkable for this card ({p:.0%} percentile)"
    add("amt_pctile_prior", AMOUNT, w, note)


def _geography(i: RubricInput, add) -> None:
    """A billing region this card has never used before.

    Rare and therefore informative: exactly one of the 20 exam cases has it (HHG-015), which is
    also the only case with three independent signals stacked. Note that a NULL `addr1` is not a
    new region - 94.8% of `C`-product transactions carry no region at all, and
    `features/baseline.add_novelty` already returns False for those rather than True.
    """
    if i.new_region:
        add("new_region", REGION, 0.90,
            "the transaction bills to a region this card has never used before")
    if i.new_product:
        add("new_product", PRODUCT, 0.40,
            "first use of this product code on the card")
    if i.new_email_domain:
        add("new_email_domain", COUNTERPARTY, 0.50,
            "the purchaser email domain is one this card has never transacted with before")
    if i.new_time_of_day:
        add("new_time_of_day", TIMING, 0.30,
            "the transaction falls in a six-hour window of the day this card has never been "
            "used in")


def _profile_fit(i: RubricInput, add) -> None:
    """Consistency as evidence in its own right.

    Without this the rubric is one-directional: a dozen families can raise suspicion and only a
    low amount can lower it, so nothing can ever be cleared without asking the cardholder. That
    is the wrong shape for a pack the brief says is roughly half legitimate, and it is not how
    the decision is actually made - an analyst clears a case when everything about the
    transaction looks like this cardholder across several independent dimensions.

    The checks deliberately EXCLUDE the amount, which already has its own family, so this stays
    independent of it. It also refuses to fire when anything genuinely novel or connected is
    present: consistency is only exonerating in the absence of an anomaly, never in spite of one.
    """
    if i.n_prior_txns < 20:
        return                      # too little history for "normal for this card" to mean much
    if (i.new_region or i.new_product or i.new_email_domain or i.new_time_of_day
            or i.ring_signal or i.card_testing or i.proxy in ("ANONYMOUS", "HIDDEN")
            or i.step_up_result == "failed"):
        return

    checks: list[str] = []
    if i.known_region:
        checks.append("bills to a region the card already uses")
    if i.known_product:
        checks.append("uses a product code the card already uses")
    if i.known_email_domain:
        checks.append("transacts with an email domain the card already uses")
    if i.device_state == "Found":
        checks.append("runs on a device the account has been seen on before")
    if i.timing_typical:
        checks.append("falls in the card's usual time of day")
    if i.prior_txns_1h <= 1 and i.prior_txns_24h <= 3:
        checks.append("arrives at the card's normal pace, with no burst around it")

    if len(checks) >= 5:
        add("profile_fit", FIT, -1.30,
            f"nothing about this transaction is new for this card: it {', '.join(checks)}")
    elif len(checks) >= 3:
        add("profile_fit", FIT, -0.90,
            f"the transaction is consistent with this card on {len(checks)} independent "
            f"dimensions: it {', '.join(checks)}")


def _device(i: RubricInput, add) -> None:
    """Device and connection signals, weighted by how common they actually are.

    `id_15 = New` covers 43% of all identity records and 11 of the 20 exam cases. It cannot be
    what separates them, so it gets a weight below MATERIAL and can never be the second
    independent signal. A proxy (`id_23`) appears on 3.8% of identity records, so it can.
    """
    if i.device_state == "New":
        add("device_state_new", DEVICE, 0.15,
            "device is new to this account, which is true of 43% of all device records and is "
            "not evidence on its own")
    elif i.device_state == "Found":
        add("device_state_found", DEVICE, -0.10,
            "the device is one this account has been seen on before")

    if i.proxy in ("ANONYMOUS", "HIDDEN"):
        article = "an" if i.proxy[0] in "AEIOU" else "a"
        add("proxy", DEVICE, 0.90,
            f"the connection is behind {article} {i.proxy.lower()} proxy, present on under 2% of "
            "device records")
    elif i.proxy == "TRANSPARENT":
        add("proxy", DEVICE, 0.30, "the connection reports a transparent proxy")


def _shared_origin(i: RubricInput, add) -> None:
    """A ring signal is a claim about other people's cards, so the bar is high.

    Only fires after the calibrated filter in `features/rings.py` (full profile, <= 10 global
    cards, >= 2 cards in window, <= 14-day span), which cuts Nov-Dec candidates from 1,671 to
    266. The remaining known failure mode is volume: a card with thousands of transactions
    touches many devices, which put HHG-018 in nine "rings" and HHG-007 in four. When every card
    on the profile is high-volume the link is nearly worthless, so it is recorded and discounted
    rather than dropped - the agent should still see it.
    """
    if not i.ring_signal:
        return
    if i.ring_volume_artefact:
        add("shared_device_profile", RING, 0.25,
            f"a specific device profile links {i.ring_n_cards} cards, but every card on it is "
            "high-volume, so the link is probably an artefact of transaction count")
    elif i.ring_strength == "moderate":
        # The profile is a common-ish device model, so the link is real but not decisive on its
        # own - HHG-014's profile sits on 52 cards book-wide. It still earns a material
        # contribution, because three unrelated cardholders inside a fortnight is a pattern.
        add("shared_device_profile", RING, 0.80,
            f"a fully specified device profile links {i.ring_n_cards} cards across "
            f"{i.ring_n_customers} different customers inside a 14-day window; the profile "
            f"appears on {i.ring_profile_cards} cards book-wide, so it is a real link rather "
            "than a rare one")
    else:
        add("shared_device_profile", RING, 1.60,
            f"a rare, fully specified device profile links {i.ring_n_cards} cards across "
            f"{i.ring_n_customers} different customers inside a 14-day window")


def _sequence(i: RubricInput, add) -> None:
    if i.card_testing:
        add("card_testing_sequence", TESTING, 1.80,
            "a card-testing sequence matching policy R5 precedes this transaction: three or "
            "more small online authorisations within an hour")
        if i.card_testing_escalated:
            add("card_testing_cleared_over_100", TESTING, 0.50,
                "a purchase over $100 has already cleared after the testing burst")
    if i.prior_txns_1h >= 3:
        add("velocity_1h", SEQUENCE, 0.70,
            f"{i.prior_txns_1h} prior authorisations on this card in the preceding hour")
    elif i.prior_txns_24h >= 8:
        add("velocity_24h", SEQUENCE, 0.35,
            f"{i.prior_txns_24h} prior authorisations on this card in the preceding 24 hours")


def _risk_model(i: RubricInput, add) -> None:
    """The bank's own score. The brief calls it an input, not a verdict, and the data agrees.

    Mean 0.171 across the book; 2.9% of rows exceed 0.7. Six of the eight customer-report exam
    cases sit below 0.40, one of them a $482 dispute at 0.25. So the score nudges and never
    decides - its largest possible contribution (0.35) is below MATERIAL by construction, which
    means it can never be one of the two independent signals that let the probability leave the
    uncertainty band.
    """
    s = i.risk_score
    if s is None:
        return
    if s >= 0.90:
        add("risk_score", MODEL, 0.35,
            f"the bank's model scores this transaction {s:.2f}, in the top 0.3% of the book")
    elif s >= 0.70:
        add("risk_score", MODEL, 0.20,
            f"the bank's model scores this transaction {s:.2f}, in the top 2.9% of the book")
    elif s <= 0.20:
        add("risk_score", MODEL, -0.10,
            f"the bank's model scores this transaction {s:.2f}, below its median of 0.12")
    else:
        add("risk_score", MODEL, 0.0,
            f"the bank's model scores this transaction {s:.2f}, close to the book mean of 0.17")


def _customer(i: RubricInput, add) -> None:
    """What the cardholder says, and what the trigger already implies they said.

    A customer-report trigger already means the customer disputes the transaction, so R2 applies
    without asking again - the weight is the same whether the dispute arrived as the trigger or
    as a response. R7 is the brake: a disputed charge that matches the cardholder's own recurring
    pattern is a customer who forgot a subscription, not fraud, and the policy says explicitly
    not to block.
    """
    disputes = i.trigger_type == "customer_report" or i.customer_response == "denied"

    if i.customer_response == "confirmed":
        add("customer_confirms", CUSTOMER, -2.50,
            "the cardholder confirms making the transaction, which settles the question")
    elif disputes:
        add("customer_disputes", CUSTOMER, 0.80,
            "the cardholder does not recognise the transaction"
            + (" (arrived as the trigger)" if i.trigger_type == "customer_report" else ""))
    elif i.customer_response == "no_reply":
        add("customer_no_reply", CUSTOMER, 0.15,
            "no reply from the cardholder within the policy's 24-hour window, which is weak "
            "evidence in either direction")

    if i.matches_recurring_pattern:
        add("matches_recurring_pattern", CUSTOMER, -1.80,
            "the disputed charge matches this cardholder's own recurring pattern - same "
            "merchant, comparable amount, regular interval (policy R7)")

    if i.step_up_result == "failed":
        add("step_up_failed", CUSTOMER, 2.00,
            "step-up authentication was requested and failed")
    elif i.step_up_result == "passed":
        add("step_up_passed", CUSTOMER, -1.50,
            "step-up authentication was requested and passed, so whoever holds the card also "
            "holds the second factor")


def _history(i: RubricInput, add) -> None:
    """Prior closed cases on this customer, as-of only.

    Deliberately kept BELOW MATERIAL, for the same reason as `id_15 = New`. 16 of the 20 exam
    customers already have closed-case history and that history is 83.8% confirmed fraud, so
    "this customer has prior fraud" is closer to a property of being in the dataset at all than a
    property of this transaction. It corroborates a case built on transaction-level evidence; it
    must never be the second independent signal that licenses a decisive verdict. A cleared
    history pulls the other way, and just as weakly.
    """
    if i.prior_confirmed_fraud >= 5:
        add("prior_confirmed_fraud", HISTORY, 0.35,
            f"{i.prior_confirmed_fraud} prior closed cases on this customer were confirmed fraud")
    elif i.prior_confirmed_fraud >= 1:
        add("prior_confirmed_fraud", HISTORY, 0.20,
            f"{i.prior_confirmed_fraud} prior closed case(s) on this customer were confirmed "
            "fraud")
    elif i.prior_cleared >= 2:
        add("prior_cleared", HISTORY, -0.20,
            f"{i.prior_cleared} prior investigations on this customer were cleared with no "
            "fraud found")


_PARTS = (_amount, _geography, _profile_fit, _device, _shared_origin, _sequence, _risk_model,
          _customer, _history)


def score(i: RubricInput, prior: float = BASE_PRIOR) -> RubricScore:
    """Turn observations into a probability, its drivers, and its independence count.

    The band clamp at the end is policy section 6 expressed as code: a decisive probability
    resting on one family of evidence is not decisive, it is overconfident, and the agent should
    go and get a second signal instead of acting.
    """
    contributions: list[Contribution] = []

    def add(signal: str, family: str, log_odds: float, note: str) -> None:
        contributions.append(Contribution(signal, family, log_odds, note))

    for part in _PARTS:
        part(i, add)

    total = _logit(prior) + sum(c.log_odds for c in contributions)
    p = _sigmoid(total)

    result = RubricScore(probability=p, contributions=contributions, prior=prior)

    # R3 is dispositive unless the account itself looks compromised - see CONFIRMED_CAP.
    ato_indicators = i.ring_signal or i.card_testing or i.step_up_result == "failed"
    if i.customer_response == "confirmed" and not ato_indicators:
        result.probability = min(p, CONFIRMED_CAP)
        result.override = ("R3: the cardholder confirms the transaction, which is dispositive "
                           "and cannot be out-voted by anomaly signals")
        result.override_verdict = "legitimate"
        return result

    # A DISPUTE MAY ONLY BE OVERRIDDEN BY AN EXPLANATION.
    #
    # When the cardholder says the transaction is not theirs and the transaction nevertheless
    # looks entirely ordinary for the card, the weighted sum lands on `legitimate` - and the case
    # file then says "no fraud found" about a charge the account holder is disputing, while the
    # policy engine simultaneously blocks the card under R2. That is not a confident finding, it
    # is two pieces of evidence pointing opposite ways.
    #
    # So a dispute can be reasoned away only by something that explains it: the cardholder's own
    # recurring pattern (R7 - they forgot a subscription), or the cardholder withdrawing it.
    # Absent an explanation the verdict is held at `uncertain`, which is what sends the case to a
    # human under R8 rather than closing it against the account holder's word.
    disputes = i.trigger_type == "customer_report" or i.customer_response == "denied"
    if (disputes and p <= LEGITIMATE_AT and not i.matches_recurring_pattern
            and i.customer_response != "confirmed"):
        result.override = ("The cardholder disputes this transaction and nothing in the evidence "
                           "explains the dispute, so the assessment is held at uncertain rather "
                           "than closed against their statement")
        result.override_verdict = "uncertain"
        return result

    if result.n_independent_signals < 2 and not (BAND_LOW <= p <= BAND_HIGH):
        result.probability = min(max(p, BAND_LOW), BAND_HIGH)
        result.clamped = True
    return result
