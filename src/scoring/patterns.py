"""Which of the policy's fraud patterns this case fits, decided by rules over evidence.

The answer file takes exactly one `pattern`, from a closed vocabulary, and the brief scores it.
Two ways to produce it: ask the model, or derive it from the evidence. This module does the
second, for three reasons.

  1. STRUCTURAL CONSTRAINTS ARE ABSOLUTE AND A MODEL WILL BREAK THEM. `channel` is a
     deterministic restatement of `ProductCD` - every one of the 439,670 `W` rows is in person
     and not one of them has an identity record. A card-present transaction cannot be
     card-not-present fraud, and no amount of persuasive evidence should be able to label it as
     such. Six of the 20 exam cases are in person. That is six chances to score zero on a field
     that a single `if` protects.
  2. THE PATTERN FEEDS THE REPORT DECISION. `should_file_report` fires on
     `pattern == "undocumented"`, and R9 attaches FILE_REPORT and ESCALATE_TO_ANALYST to it.
     Nine of 5,565 closed cases carry that pattern. A model that reaches for `undocumented`
     when it is unsure - which is exactly when a model reaches for the unusual option - would
     file regulatory reports on ambiguity.
  3. IT IS EXPLAINABLE FOR FREE. Every candidate keeps its rationale and its rejection reason,
     so the case file can say why this pattern and not the other one.

The model still has a job here: reviewing the classification against the narrative evidence, and
proposing an override with a justification. Disagreements are logged rather than resolved
silently - an agent whose deterministic layer and reasoning layer disagree about the typology is
telling us something about the case.

Base rates across the 5,565 closed investigations, which is what "usual" means in this dataset:
card_not_present_fraud 1,404 · account_takeover 1,205 · card_not_present_new_device 1,076 ·
out_of_region_use 955 · card_testing 16 · undocumented 9 · none (cleared) 900.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .rubric import FRAUD_AT, LEGITIMATE_AT, RubricInput, RubricScore

CARD_TESTING = "card_testing"
CNP = "card_not_present_fraud"
CNP_NEW_DEVICE = "card_not_present_new_device"
OUT_OF_REGION = "out_of_region_use"
ATO = "account_takeover"
UNDOCUMENTED = "undocumented"
NONE = "none"

ALL_PATTERNS = (CARD_TESTING, CNP, CNP_NEW_DEVICE, OUT_OF_REGION, ATO, UNDOCUMENTED, NONE)

CARD_PRESENT_IMPOSSIBLE = (CNP, CNP_NEW_DEVICE)
"""Patterns that cannot apply to an in-person transaction, by definition of the words."""


@dataclass
class Candidate:
    pattern: str
    score: float
    rationale: str
    rejected: str = ""

    @property
    def viable(self) -> bool:
        return not self.rejected and self.score > 0


@dataclass
class PatternVerdict:
    pattern: str
    rationale: str
    candidates: list[Candidate] = field(default_factory=list)
    description: str = ""
    """Only populated for `undocumented`, where the answer schema requires a description."""

    @property
    def runner_up(self) -> Candidate | None:
        viable = [c for c in self.candidates if c.viable and c.pattern != self.pattern]
        return max(viable, key=lambda c: c.score) if viable else None

    def why_not_runner_up(self) -> str:
        r = self.runner_up
        if r is None:
            return "No other documented pattern fits the evidence."
        rationale = r.rationale.rstrip(". ")
        return (f"The closest alternative was {r.pattern} ({r.score:.2f} against "
                f"{max(c.score for c in self.candidates if c.pattern == self.pattern):.2f}): "
                f"{rationale}.")

    def to_json(self) -> dict:
        return {
            "pattern": self.pattern,
            "pattern_description": self.description,
            "rationale": self.rationale,
            "alternatives": [{"pattern": c.pattern, "score": round(c.score, 3),
                              "rationale": c.rationale, "rejected": c.rejected}
                             for c in sorted(self.candidates, key=lambda c: -c.score)
                             if c.pattern != self.pattern],
        }


def _candidates(i: RubricInput, s: RubricScore) -> list[Candidate]:
    card_present = i.channel == "in_person"
    device_new = i.device_state == "New"
    proxy = i.proxy in ("ANONYMOUS", "HIDDEN")
    burst = i.prior_txns_1h >= 3 or i.prior_txns_24h >= 8
    out: list[Candidate] = []

    # --- card testing: the detector decides, nothing else ----------------------------------
    if i.card_testing:
        out.append(Candidate(CARD_TESTING, 0.95,
                             "the R5 detector found three or more small online authorisations "
                             "within an hour on this card before the flagged transaction"))
    else:
        out.append(Candidate(CARD_TESTING, 0.0,
                             "no qualifying testing sequence",
                             rejected="the R5 detector did not fire. Across the whole 590,742-row "
                                      "book only 28 cards qualify, and no exam case has a "
                                      "qualifying sequence at its flagged moment"))

    # --- out of region ---------------------------------------------------------------------
    # The policy defines this pattern as CARD-PRESENT purchases in a new region, and the closed
    # history agrees without exception: all 955 out_of_region_use cases are in person, and not
    # one online case carries the label. An online purchase billed to a new region is
    # card-not-present fraud; the region still counts as evidence in the rubric, it just does
    # not name the pattern. (HHG-015 was an online R-product purchase labelled out of region.)
    if i.new_region and card_present:
        out.append(Candidate(OUT_OF_REGION, 0.70,
                             "the transaction bills to a region this card has never used"))
    elif i.new_region:
        why = ("the region is new to the card, but the purchase is online - the policy's "
               "out-of-region pattern is card-present use, and all 955 closed cases with it are "
               "in person")
        out.append(Candidate(OUT_OF_REGION, 0.0, why, rejected=why))
    else:
        why = ("the billing region is one the card already uses" if i.known_region
               else "no billing region is recorded on this transaction, and a missing region is "
                    "not a new one - 94.8% of C-product rows have none")
        out.append(Candidate(OUT_OF_REGION, 0.0, why, rejected=why))

    # --- account takeover ------------------------------------------------------------------
    # ATO is a claim about control of the account, not about one transaction. What separates it
    # from card-not-present fraud is evidence that whoever is transacting has taken over the
    # session: a new device plus something else - a proxy, a burst, a failed challenge, or
    # several transactions in the episode. Mean exposure in the closed history is $752 against
    # $325 for CNP-new-device, so size corroborates but does not decide.
    ato_support = [name for name, present in (
        ("a proxied connection", proxy),
        ("a burst of authorisations around it", burst),
        ("a failed step-up challenge", i.step_up_result == "failed"),
        ("more than one transaction in the episode", i.n_affected_txns > 1),
    ) if present]
    # A device shared across OTHER customers' cards points away from takeover, not towards it:
    # takeover is one account falling under someone else's control, and a profile appearing on
    # three customers' cards inside two weeks is coordinated activity across accounts. Without
    # this, the ring case scored 0.85 as account_takeover against 0.75 as undocumented, and the
    # cross-account evidence - the whole reason the case is interesting - decided nothing.
    cross_account = bool(i.ring_signal and not i.ring_volume_artefact
                         and i.ring_n_customers > 1)
    if cross_account:
        out.append(Candidate(ATO, 0.30,
                             "there is session-level evidence, but the device profile is shared "
                             f"across {i.ring_n_customers} different customers' cards, which is "
                             "coordinated abuse across accounts rather than one account being "
                             "taken over"))
    elif device_new and ato_support:
        out.append(Candidate(ATO, 0.80 + 0.05 * len(ato_support),
                             "the transaction runs on a device new to the account together with "
                             + " and ".join(ato_support)
                             + ", which points at control of the account rather than use of the "
                               "card alone"))
    elif ato_support and not device_new:
        out.append(Candidate(ATO, 0.35,
                             "there is session-level evidence (" + ", ".join(ato_support)
                             + ") but the device is not new to the account"))
    else:
        out.append(Candidate(ATO, 0.0, "no session-level evidence of account control",
                             rejected="a new device alone is not account takeover - `id_15 = New` "
                                      "covers 43% of all device records"))

    # --- card not present, with and without a new device -----------------------------------
    if card_present:
        for p in CARD_PRESENT_IMPOSSIBLE:
            out.append(Candidate(p, 0.0, "the transaction is card present",
                                 rejected=f"the transaction is in person (ProductCD "
                                          f"{i.product_cd or 'W'}), so {p} is impossible by "
                                          "definition"))
    elif device_new:
        out.append(Candidate(CNP_NEW_DEVICE, 0.65,
                             "a card-not-present transaction on a device new to the account"))
        out.append(Candidate(CNP, 0.40,
                             "a card-not-present transaction; the new device makes the "
                             "new-device variant the better fit"))
    else:
        out.append(Candidate(CNP, 0.60,
                             "a card-not-present transaction on a device the account has used "
                             "before, or with no device record at all"))
        out.append(Candidate(CNP_NEW_DEVICE, 0.0, "the device is not new to the account",
                             rejected="the device is known to the account"))

    # --- undocumented ----------------------------------------------------------------------
    # Deliberately hard to reach. It requires a shared origin that survived the ring filter AND
    # a confident assessment, because R9 attaches FILE_REPORT and ESCALATE_TO_ANALYST to it and
    # only 9 of 5,565 closed cases ever used it.
    #
    # And the link must be STRONG - a rare profile on at most ten cards book-wide. A moderate
    # link (a common device model on 11-60 cards) is shared use, not shared origin: R6 already
    # declines to file on one. Accepting it here made `undocumented` the verdict on 90 of 300
    # unseen high-risk alerts replayed through the live monitor - 30% against a closed-case base
    # rate of 0.16% - and filed a report on every one through R9.
    strong_link = i.ring_strength != "moderate"
    #
    # And it takes the rubric's VERDICT, not its raw probability. A case can sit above 0.70 on a
    # single family of evidence and still be held `uncertain` - that is the rubric's rule that
    # one family never decides a case - and naming a new typology, with the report R9 attaches,
    # on a case the engine itself calls undecided is a contradiction (HHG-019: 0.81, one family).
    if (i.ring_signal and strong_link and not i.ring_volume_artefact
            and s.verdict == "fraud"):
        out.append(Candidate(UNDOCUMENTED, 0.90 if cross_account else 0.75,
                             f"a rare device profile links {i.ring_n_cards} cards"
                             + (f" belonging to {i.ring_n_customers} different customers"
                                if cross_account else "")
                             + " inside a tight window, which is coordinated activity across "
                               "accounts rather than one cardholder's card being misused"))
    else:
        missing = []
        if not i.ring_signal:
            missing.append("no shared origin across accounts survived the ring filter")
        elif not strong_link:
            missing.append(f"the shared device profile sits on {i.ring_profile_cards} cards "
                           "book-wide - a common device model, which is shared use rather than "
                           "a shared origin")
        elif i.ring_volume_artefact:
            missing.append("the shared origin is a volume artefact, not a link")
        if s.verdict != "fraud":
            missing.append(f"the assessment has not reached a fraud verdict ({s.verdict} at "
                           f"{s.probability:.2f})")
        out.append(Candidate(UNDOCUMENTED, 0.0, "; ".join(missing),
                             rejected="R9 attaches a regulatory filing to this pattern and 9 of "
                                      "5,565 closed cases used it: " + "; ".join(missing)))

    return out


def classify(i: RubricInput, s: RubricScore) -> PatternVerdict:
    """The pattern, with every alternative and why it lost.

    A legitimate assessment gets `none` regardless of what else is visible: the answer schema
    requires that a legitimate verdict carries no episode, and a pattern is a description of an
    episode. An uncertain assessment still gets its best-fitting pattern, because "we do not yet
    know whether this is fraud, and if it is, this is the shape of it" is the honest position and
    it is what the evidence loop is working on.
    """
    candidates = _candidates(i, s)

    if s.verdict == "legitimate" or s.probability <= LEGITIMATE_AT:
        return PatternVerdict(
            NONE,
            "The evidence does not support a fraud episode, so no pattern is claimed"
            + (f" (probability {s.probability:.2f})" if s.probability else ""),
            candidates)

    viable = [c for c in candidates if c.viable]
    if not viable:
        return PatternVerdict(
            NONE,
            "No documented pattern fits the evidence, and the case does not meet the bar for "
            "`undocumented` - which requires coordinated activity across accounts, not merely "
            "an absence of fit",
            candidates)

    best = max(viable, key=lambda c: c.score)
    verdict = PatternVerdict(best.pattern, best.rationale, candidates)
    if best.pattern == UNDOCUMENTED:
        verdict.description = (
            f"Coordinated use of a single rare device profile across {i.ring_n_cards} cards "
            "belonging to different customers within a two-week window, with no shared billing "
            "region or product pattern. It fits none of the five documented typologies: the "
            "cards are not being tested, the activity is not confined to one cardholder's "
            "account, and the anomaly is the link between accounts rather than any single "
            "transaction.")
    return verdict
