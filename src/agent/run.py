"""The investigation, end to end, with no LLM in the loop.

    python -m src.agent.run --data data_fixture --out cases_fixture
    python -m src.agent.run --data data --out cases

WHY THE DETERMINISTIC RUN COMES FIRST. The deliverable is 20 answer files. This module produces
all 20 without an API key, without a graph endpoint and without a network, which means the
submission is never one outage away from zero. Everything the LLM will add - the prose, the
typology judgement, the narrative - is an improvement on a working artifact rather than the thing
holding it up. It is also the only way to iterate on the investigation logic quickly: a full
20-case run here takes about a second.

THE STAGES map onto the brief's loop, and each is a function below:

    trigger -> investigate -> gather evidence -> assess uncertainty
             -> [request evidence -> reassess]*  -> take action -> explain -> update memory

Two things the brief scores that are easy to get wrong, and are structural here rather than
reconstructed afterwards:

  - `next_best_actions.initial` is captured on the FIRST pass through Take Action, before any
    evidence request is issued, and `final` on the LAST. They are two separate compositions of
    two different policy contexts, not one list edited in place.
  - Routes on both lists are resolved at the case's FINAL exposure, because that is what the
    approval table takes as input and what the validator checks against. The actions in `initial`
    are the ones that were recommended first; only their routing is settled once.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..features.recurring import Recurrence
from ..graph.local import _region_code as _region
from ..graph.interface import GraphBackend
from ..graph.local import LocalBackend
from ..io.answer import Answer, Case, Evidence, EvidenceRequest, SAR
from ..io.ledger import load_ledger
from ..policy.actions import Action, Decision, compose
from ..policy.rules import PolicyContext, evaluate, should_stop
from ..scoring.patterns import CARD_TESTING, NONE, PatternVerdict, classify
from ..scoring.rubric import BASE_PRIOR, RubricInput, RubricScore, score
from .context import build_policy_context
from .meter import Meter, roll_up
from .simulate import respond
from .voi import evaluate_options

MAX_EVIDENCE_LOOPS = 3
"""Policy section 6. On cap-out the verdict stays `uncertain` and R8 decides what happens next."""


# --------------------------------------------------------------------------------------------
# What one case looks like while it is being worked on
# --------------------------------------------------------------------------------------------

@dataclass
class CaseState:
    case_id: str
    customer_id: str
    card_id: str | None
    card_key: str
    flagged_txn_id: str
    trigger_type: str
    narrative: str
    opened_at: pd.Timestamp

    rubric_input: RubricInput = field(default_factory=RubricInput)
    evidence: list[Evidence] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    requests: list[EvidenceRequest] = field(default_factory=list)

    initial_pairs: list[tuple[Action, str]] = field(default_factory=list)
    final_pairs: list[tuple[Action, str]] = field(default_factory=list)
    initial_json: list[dict] = field(default_factory=list)

    ring_element: str = ""
    connected_card_ids: list[str] = field(default_factory=list)
    connected_profiles: list[str] = field(default_factory=list)
    similar_cases: list[str] = field(default_factory=list)
    episode_txn_ids: list[str] = field(default_factory=list)
    n_cards: int = 1
    recurrence: Recurrence | None = None
    voi_log: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    loops: int = 0

    def note(self, claim: str, source: str, ref: str, entity_ids: list[str],
             meter: Meter) -> None:
        """Append an evidence item. Same shape as the answer contract, so serialising is a copy."""
        self.evidence.append(Evidence(claim=claim, source=source, ref=ref,
                                      entity_ids=[str(e) for e in entity_ids]))
        meter.record_evidence(source)
        meter.event("evidence", source=source, ref=ref, claim=claim)

    def decide(self, text: str) -> None:
        self.decisions.append(text)


# --------------------------------------------------------------------------------------------
# Stage 1 + 2: investigate and gather evidence
# --------------------------------------------------------------------------------------------

def fetch(be: GraphBackend, meter: Meter, calls: dict[str, tuple]) -> dict[str, Any]:
    """Run independent graph reads, concurrently where the backend is remote.

    Against TigerGraph every read is an HTTPS round trip, and the reads in one wave of `gather`
    depend on nothing but the case pack - so nine sequential round trips become two waves. The
    parquet backend runs in-process under the GIL and gains nothing from threads, so it keeps
    running them in order, which also keeps the offline run exactly reproducible.
    """
    def one(name: str):
        fn, args, kwargs = calls[name]
        return meter.measure(f"graph.{name}")(fn)(*args, **kwargs)

    if not getattr(be, "concurrent_reads", False) or len(calls) == 1:
        return {name: one(name) for name in calls}
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {name: pool.submit(one, name) for name in calls}
        return {name: f.result() for name, f in futures.items()}


def gather(be: GraphBackend, state: CaseState, meter: Meter) -> None:
    """Pull every piece of evidence the closed query set can produce, and log each one.

    Two waves. The first needs only what the case pack already says - the transaction, the
    customer, the card and the time - so all seven reads go out together. The second needs the
    channel and amount the first wave returned. The evidence is then written up in a fixed
    order, so the case record reads the same however the reads were scheduled.
    """
    wave = fetch(be, meter, {
        "get_transaction": (be.get_transaction, (state.flagged_txn_id,), {}),
        "get_customer": (be.get_customer, (state.customer_id,), {}),
        "card_baseline": (be.card_baseline, (state.card_key, state.flagged_txn_id), {}),
        "recurring_charge": (be.recurring_charge, (state.card_key, state.flagged_txn_id), {}),
        "card_testing_episode": (be.card_testing_episode, (state.card_key, state.opened_at), {}),
        "ring_signals": (be.ring_signals, (state.card_key, state.opened_at), {}),
        "prior_cases_for_customer": (be.prior_cases_for_customer,
                                     (state.customer_id, state.opened_at), {}),
    })
    txn = wave["get_transaction"]
    amount = float(txn["TransactionAmt"])
    channel = str(txn["channel"])
    risk = float(txn["risk_score"])

    second = {"similar_closed_cases": (be.similar_closed_cases, (),
                                       {"as_of": state.opened_at, "channel": channel,
                                        "exposure_usd": amount, "k": 3})}
    if channel == "online":
        second["device_for_transaction"] = (be.device_for_transaction,
                                            (state.flagged_txn_id,), {})
    wave.update(fetch(be, meter, second))

    state.note(
        f"The flagged transaction is a {channel.replace('_', '-')} authorisation of "
        f"${amount:,.2f} on {str(txn['ts'])[:16]} under product code {txn['ProductCD']}, "
        f"scored {risk:.2f} by the bank's model.",
        "graph", "query:get_transaction", [state.flagged_txn_id], meter)

    if state.trigger_type == "customer_report":
        state.note(f"The cardholder reported this transaction as unrecognised: {state.narrative}",
                   "customer", "case_pack.trigger", [state.flagged_txn_id], meter)
    else:
        state.note(f"The investigation was triggered by {state.trigger_type.replace('_', ' ')}: "
                   f"{state.narrative}",
                   "external", "case_pack.trigger", [state.flagged_txn_id], meter)

    customer = wave["get_customer"]
    state.n_cards = int(customer["n_cards"])

    baseline = wave["card_baseline"]
    if baseline.amt_pctile_prior is not None and baseline.n_prior_txns:
        state.note(
            f"Against this card's own prior history of {baseline.n_prior_txns:,} transactions, "
            f"${amount:,.2f} sits at the {baseline.amt_pctile_prior:.0%} percentile "
            f"(mean ${baseline.amt_mean_prior or 0:,.2f}, largest previously "
            f"${baseline.amt_max_prior or 0:,.2f}).",
            "graph", "query:card_baseline", [state.flagged_txn_id], meter)
    if baseline.new_region:
        state.note(
            f"Billing region {_region(txn.get('addr1'))} has never appeared on this card "
            f"before; it has "
            f"previously used {', '.join(baseline.known_regions) or 'no recorded region'}.",
            "graph", "query:card_baseline", [state.flagged_txn_id], meter)
    if baseline.prior_txns_1h or baseline.prior_txns_24h:
        state.note(
            f"{baseline.prior_txns_1h} prior authorisation(s) on this card in the preceding hour "
            f"and {baseline.prior_txns_24h} in the preceding 24 hours.",
            "graph", "query:card_baseline", [state.flagged_txn_id], meter)

    device = None
    if channel == "online":
        device = wave["device_for_transaction"]
        if device:
            state.note(
                f"The transaction ran on device profile `{device['device_profile']}`, reported as "
                f"{device.get('device_state') or 'unknown'} to this account"
                + (f", behind a {device['proxy']} proxy" if device.get("proxy") else "")
                + f". That profile appears on {device.get('global_card_count')} card(s) "
                  "book-wide.",
                "graph", "query:device_for_transaction",
                [state.flagged_txn_id, device["device_profile"]], meter)
            state.connected_profiles = [device["device_profile"]]
        else:
            state.note("No device or connection record exists for this transaction, which is "
                       "expected for some online product codes and is not itself a signal.",
                       "graph", "query:device_for_transaction", [state.flagged_txn_id], meter)
    else:
        state.note("The transaction is card present, so no device or connection record exists - "
                   "true of every in-person transaction in the book, and not a signal.",
                   "graph", "query:device_for_transaction", [state.flagged_txn_id], meter)

    recurrence = wave["recurring_charge"]
    state.recurrence = recurrence
    state.note(recurrence.as_evidence_claim(), "graph", "query:recurring_charge",
               [state.flagged_txn_id, *recurrence.prior_txn_ids], meter)

    episode = wave["card_testing_episode"]
    if episode:
        state.episode_txn_ids = [str(t) for t in episode["small_txn_ids"]]
        state.note(
            f"A card-testing sequence matching policy R5 precedes this transaction: "
            f"{episode['n_small']} authorisations under $5 between {episode['window'][0][:16]} "
            f"and {episode['window'][1][:16]}"
            + (f", followed by a purchase of ${episode['max_follow_amt']:,.2f} that cleared."
               if episode["escalate"] else "."),
            "graph", "query:card_testing_episode",
            [state.flagged_txn_id, *state.episode_txn_ids], meter)

    rings = wave["ring_signals"]
    ring = next((r for r in rings if not r.volume_artefact_risk), rings[0] if rings else None)
    if ring:
        state.ring_element = ring.element
        state.connected_card_ids = [c for c in ring.card_ids if c != state.card_id]
        state.connected_profiles = sorted(set(state.connected_profiles) | {ring.device_profile})
        state.note(
            f"{ring.element} links {ring.n_cards} cards across {ring.n_customers} customers "
            f"within {ring.span_days:.1f} days, and appears on only "
            f"{ring.global_card_count} cards book-wide"
            + (" - but every card on it is high-volume, so the link may be an artefact of "
               "transaction count rather than a shared origin."
               if ring.volume_artefact_risk else "."),
            "graph", "query:ring_signals",
            [state.flagged_txn_id, ring.device_profile, *ring.card_ids], meter)

    prior = wave["prior_cases_for_customer"]
    n_fraud = int((prior.outcome == "confirmed_fraud").sum()) if len(prior) else 0
    n_cleared = int((prior.outcome == "cleared").sum()) if len(prior) else 0
    if len(prior):
        state.note(
            f"This customer has {len(prior)} closed investigation(s) before this alert: "
            f"{n_fraud} confirmed fraud, {n_cleared} cleared. Patterns previously found: "
            f"{', '.join(sorted(set(prior.pattern) - {'none'})) or 'none'}.",
            "graph", "query:prior_cases_for_customer",
            [state.customer_id, *prior.case_id.tolist()], meter)
    else:
        state.note("This customer has no closed investigations before this alert.",
                   "graph", "query:prior_cases_for_customer", [state.customer_id], meter)

    similar = wave["similar_closed_cases"]
    if len(similar):
        state.similar_cases = similar.case_id.tolist()
        top = similar.iloc[0]
        state.note(
            f"The closest prior investigations by channel, exposure and recency are "
            f"{', '.join(state.similar_cases)}. The nearest, {top.case_id}, was a "
            f"${float(top.exposure_usd):,.2f} {top.channel or channel} case closed as "
            f"{top.outcome}"
            + (f" with pattern {top.pattern}." if top.pattern != "none" else "."),
            "graph", "query:similar_closed_cases", state.similar_cases, meter)

    state.rubric_input = RubricInput(
        channel=channel,
        product_cd=str(txn["ProductCD"]),
        n_affected_txns=1 + len(state.episode_txn_ids),
        exposure_usd=amount,
        amt_pctile_prior=baseline.amt_pctile_prior,
        n_prior_txns=baseline.n_prior_txns,
        amount_usd=amount,
        new_region=baseline.new_region,
        known_region=bool(txn.get("addr1") is not None
                          and not baseline.new_region and baseline.known_regions),
        new_product=baseline.new_product,
        known_product=str(txn["ProductCD"]) in baseline.known_products,
        device_state=(device or {}).get("device_state"),
        proxy=(device or {}).get("proxy"),
        device_profile_is_full=bool((device or {}).get("is_full_profile")),
        device_profile_global_cards=(device or {}).get("global_card_count"),
        ring_signal=ring is not None,
        ring_volume_artefact=bool(ring.volume_artefact_risk) if ring else False,
        ring_n_cards=ring.n_cards if ring else 0,
        ring_n_customers=ring.n_customers if ring else 0,
        ring_strength=ring.strength if ring else "strong",
        ring_profile_cards=ring.global_card_count if ring else 0,
        prior_txns_1h=baseline.prior_txns_1h,
        prior_txns_24h=baseline.prior_txns_24h,
        card_testing=episode is not None,
        card_testing_escalated=bool(episode and episode["escalate"]),
        risk_score=risk,
        trigger_type=state.trigger_type,
        matches_recurring_pattern=recurrence.matches,
        prior_confirmed_fraud=n_fraud,
        prior_cleared=n_cleared,
    )


# --------------------------------------------------------------------------------------------
# Stage 3-5: assess, request evidence, reassess
# --------------------------------------------------------------------------------------------

def exposure_for(state: CaseState, verdict: str, amount: float) -> tuple[list[str], float]:
    """The transactions this case holds the bank exposed on, and their total.

    A legitimate verdict carries no episode - the schema requires it and so does the meaning of
    the word. A card-testing verdict carries the whole burst, because the episode is the
    sequence, not the single authorisation that tripped the alert.
    """
    if verdict == "legitimate":
        return [], 0.0
    ids = [state.flagged_txn_id]
    if state.rubric_input.card_testing:
        ids = sorted(set(ids) | set(state.episode_txn_ids))
    return ids, 0.0     # total is recomputed from the dataset by the caller


def context_for(state: CaseState, s: RubricScore, pattern: PatternVerdict,
                exposure: float, *, evidence_requested: bool) -> PolicyContext:
    return build_policy_context(
        state.rubric_input, s,
        exposure_usd=exposure,
        pattern=pattern.pattern,
        ring_element=state.ring_element,
        connected_card_ids=state.connected_card_ids,
        customer_n_cards=state.n_cards,
        evidence_requested=evidence_requested,
    )


def _assessment_event(s: RubricScore, pattern: PatternVerdict) -> dict:
    return {"p": round(s.probability, 3), "verdict": s.verdict, "pattern": pattern.pattern,
            "families": list(s.families), "n_signals": s.n_independent_signals}


def assess_and_act(be: GraphBackend, state: CaseState, meter: Meter,
                   amounts: dict[str, float]) -> tuple[RubricScore, PatternVerdict, float]:
    """Score, recommend, decide whether to ask for more, and recommend again.

    Returns the FINAL assessment. `state.initial_pairs` holds the first recommendation and
    `state.final_pairs` the last, so the two are produced by two separate passes rather than by
    editing one list - which is what the brief means by recording the next best action before
    and after an evidence request.
    """
    def evaluate_now(requested: bool):
        s = score(state.rubric_input)
        pattern = classify(state.rubric_input, s)
        ids, _ = exposure_for(state, s.verdict, state.rubric_input.amount_usd)
        exposure = round(sum(abs(amounts[str(t)]) for t in ids if str(t) in amounts), 2)
        ctx = context_for(state, s, pattern, exposure, evidence_requested=requested)
        return s, pattern, exposure, ctx

    with meter.stage("assess"):
        s, pattern, exposure, ctx = evaluate_now(False)
        decision = evaluate(ctx)
        state.initial_pairs = [(Action(r.action), r.reason) for r in decision.recommendations]
        meter.event("assessment", label="initial", **_assessment_event(s, pattern))
        meter.event("recommendation", label="initial", actions=decision.action_names)
        state.decide(
            f"Initial assessment: probability {s.probability:.2f} on "
            f"{s.n_independent_signals} independent signal(s) "
            f"({', '.join(s.families) or 'none'}); pattern {pattern.pattern}; recommendation "
            f"{', '.join(decision.action_names)}.")
        if s.override:
            state.decide(f"Override applied: {s.override}")

    # --- the evidence loop ------------------------------------------------------------------
    with meter.stage("gather_more_evidence"):
        while state.loops < MAX_EVIDENCE_LOOPS:
            # The comparison runs BEFORE the stopping rule is consulted, and is always recorded.
            # "Why was more evidence requested" and "why was it not" are the same question, and
            # the second answer is the more interesting one - a case that stops immediately
            # should still be able to show that every available question was priced and none of
            # them would have changed the recommendation. It costs nothing: the selector is
            # deterministic and makes no model calls.
            selection = evaluate_options(
                state.rubric_input,
                lambda ri, rs: context_for(state, rs, classify(ri, rs), exposure,
                                           evidence_requested=True),
                already_requested=tuple(r.type for r in state.requests))
            state.voi_log.append({"loop": state.loops + 1, "options": selection.to_json()})
            meter.event("evidence_value", loop=state.loops + 1,
                        options=[{"request_type": o["request_type"],
                                  "net_value": o["net_value"], "selected": o["selected"]}
                                 for o in selection.to_json()])

            stop, why = should_stop(ctx, state.loops, MAX_EVIDENCE_LOOPS)
            if stop:
                state.stop_reason = why
                break

            if selection.best is None:
                state.stop_reason = selection.stop_reason
                state.decide("No further evidence request would change the recommendation: "
                             + "; ".join(o.rationale() for o in selection.options))
                break

            option = selection.best
            simulated = respond(option)
            state.decide(
                f"Requested {option.request_type} because it carried the highest expected "
                f"change in the recommendation ({option.evoi:.2f} against friction "
                f"{option.friction:.2f}). {option.rationale()}")
            meter.event("request", type=option.request_type,
                        response=f"{simulated.statement} {simulated.counterfactual}")
            state.requests.append(EvidenceRequest(
                type=option.request_type,
                asked_after_step=3 + state.loops,
                assumed_response=f"{simulated.statement} {simulated.counterfactual}"))
            state.note(simulated.statement, "customer" if option.request_type != "analyst_info"
                       else "external", f"evidence_request:{option.request_type}",
                       [state.flagged_txn_id], meter)

            state.rubric_input = simulated.apply(state.rubric_input)
            state.loops += 1
            s, pattern, exposure, ctx = evaluate_now(True)
            meter.event("assessment", label=f"after {option.request_type}",
                        **_assessment_event(s, pattern))
            state.decide(
                f"Reassessed after {option.request_type}: probability {s.probability:.2f} on "
                f"{s.n_independent_signals} independent signal(s); pattern {pattern.pattern}.")
        else:
            state.stop_reason = (
                f"Evidence loop cap ({MAX_EVIDENCE_LOOPS}) reached without resolution; further "
                "requests are unlikely to change the decision.")

    if not state.stop_reason:
        state.stop_reason = should_stop(ctx, state.loops, MAX_EVIDENCE_LOOPS)[1]

    with meter.stage("take_action"):
        final_decision = evaluate(ctx)
        state.final_pairs = [(Action(r.action), r.reason) for r in final_decision.recommendations]
        meter.event("recommendation", label="final", actions=final_decision.action_names,
                    stop_reason=state.stop_reason)
        state.decide(f"Final recommendation: {', '.join(final_decision.action_names)}.")

    return s, pattern, exposure


# --------------------------------------------------------------------------------------------
# Stage 6: explain
# --------------------------------------------------------------------------------------------

def summarise(state: CaseState, s: RubricScore, pattern: PatternVerdict, exposure: float,
              final: Decision) -> str:
    """Two to six sentences. The evidence list carries the detail; this carries the decision."""
    drivers = s.drivers
    driver_text = ("; ".join(c.note for c in drivers) if drivers
                   else "no signal in the available evidence moved the assessment materially")
    verdict_text = {
        "fraud": "The evidence supports fraud",
        "legitimate": "The evidence does not support fraud",
        "uncertain": "The evidence is not sufficient to decide",
    }[s.verdict]

    lines = [
        f"{verdict_text} on this {state.trigger_type.replace('_', ' ')} alert for card "
        f"{state.card_id or state.card_key}, at a fraud probability of {s.probability:.2f} "
        f"resting on {s.n_independent_signals} independent signal(s): {driver_text}.",
    ]
    if pattern.pattern != NONE:
        lines.append(f"The activity fits {pattern.pattern.replace('_', ' ')}: "
                     f"{pattern.rationale.rstrip('. ')}. {pattern.why_not_runner_up()}")
    if state.requests:
        asked = ", ".join(r.type for r in state.requests)
        lines.append(f"Additional evidence was requested ({asked}) because it was the cheapest "
                     f"question whose answer could change the recommendation.")
    if s.conflicting:
        closing = Action.CLOSE_NO_FRAUD.value in final.action_names
        lines.append(
            "The evidence points both ways, so the recommendation to close is routed to an "
            "analyst for confirmation rather than acted on directly (R8)."
            if closing else
            "The evidence points both ways, which is why the case goes to an analyst rather "
            "than being decided here (R8).")
    lines.append(f"Recommended actions: {', '.join(final.action_names)}"
                 + (f", with exposure of ${exposure:,.2f}." if exposure else "."))
    return " ".join(lines)


def sar_narrative(state: CaseState, s: RubricScore, pattern: PatternVerdict, exposure: float,
                  dates: list[str]) -> str:
    """FinCEN structure: who, what, when, where, how, why suspicious - standing alone.

    Written to be readable without the case file, because that is the standard a suspicious
    activity report is held to. The LLM pass will improve the prose; the structure is fixed here
    so the required elements cannot go missing.
    """
    who = (f"Customer {state.customer_id}, holder of card {state.card_id or 'an unnamed card'}"
           f"{f' and {state.n_cards - 1} other card(s) on the same account' if state.n_cards > 1 else ''}.")
    what = (f"{len(state.episode_txn_ids) + 1 if state.rubric_input.card_testing else 1} "
            f"transaction(s) totalling ${exposure:,.2f} are reported as suspicious.")
    when = (f"The activity occurred between {dates[0]} and {dates[-1]}."
            if dates else "The activity occurred on a single date.")
    where = (f"The activity was {state.rubric_input.channel.replace('_', '-')}"
             + (f" and ran on device profile `{state.connected_profiles[0]}`."
                if state.connected_profiles else "."))
    how = f"The pattern identified is {pattern.pattern.replace('_', ' ')}. {pattern.rationale}."
    why = ("The assessment rests on: "
           + "; ".join(c.note for c in s.drivers)
           + f". The resulting fraud probability is {s.probability:.2f}.")
    trigger = (f"The investigation was opened following {state.trigger_type.replace('_', ' ')}."
               + (f" {state.narrative}" if state.narrative else ""))
    connected = ""
    if state.connected_card_ids:
        connected = (f" The shared origin also touches card(s) "
                     f"{', '.join(state.connected_card_ids)}, held by other customers, which is "
                     "why this filing covers activity beyond a single account.")
    return " ".join([who, what, when, where, how, why, trigger + connected])


# --------------------------------------------------------------------------------------------
# Stage 7: update case memory, then serialise
# --------------------------------------------------------------------------------------------

def status_for(verdict: str, final: Decision) -> str:
    if Action.ESCALATE_TO_ANALYST.value in final.action_names:
        return "escalated"
    if verdict == "fraud":
        return "closed_fraud"
    if verdict == "legitimate":
        return "closed_legitimate"
    return "open"


def investigate(be: GraphBackend, row: pd.Series, amounts: dict[str, float],
                timestamps: dict[str, str]) -> tuple[Answer, Meter, CaseState, dict]:
    """One case, from trigger to answer file."""
    meter = Meter(case_id=str(row.case_id))
    card = be.resolve_card(txn_id=str(row.flagged_txn_id))
    state = CaseState(
        case_id=str(row.case_id),
        customer_id=str(row.customer_id),
        card_id=card.card_id if card and card.citable else None,
        card_key=card.card_key if card else "",
        flagged_txn_id=str(row.flagged_txn_id),
        trigger_type=str(row.trigger_type),
        # the real case_pack.csv column is `trigger_text`; `narrative` was a guess from a
        # paraphrase of the brief and would have silently emptied every trigger claim
        narrative=str(getattr(row, "trigger_text", None)
                      or getattr(row, "narrative", "") or ""),
        opened_at=pd.Timestamp(row.opened_at),
    )
    meter.event("open", trigger=state.trigger_type, flagged_txn_id=state.flagged_txn_id,
                card_id=state.card_id, prior=BASE_PRIOR)

    with meter.stage("investigate"):
        gather(be, state, meter)

    s, pattern, exposure = assess_and_act(be, state, meter, amounts)

    # Both lists are routed at the final exposure - see the module docstring.
    initial = compose(state.initial_pairs, exposure)
    final = compose(state.final_pairs, exposure)

    affected, _ = exposure_for(state, s.verdict, state.rubric_input.amount_usd)
    affected = [t for t in affected if t in amounts]
    dates = sorted({timestamps[t][:10] for t in affected if t in timestamps})

    with meter.stage("explain"):
        summary = summarise(state, s, pattern, exposure, final)
        files_report = Action.FILE_REPORT.value in final.action_names
        if files_report:
            sar = SAR(file=True,
                      reason=next((r.reason for r in final.recommendations
                                   if r.action is Action.FILE_REPORT), "policy 3a"),
                      narrative=sar_narrative(state, s, pattern, exposure, dates),
                      subjects=[state.customer_id],
                      total_amount_usd=exposure,
                      activity_dates=[dates[0], dates[-1]] if dates else [])
        else:
            sar = SAR.not_filed(
                "No regulatory filing is required: policy 3a needs confirmed or strongly "
                "suspected fraud together with an aggravating condition (exposure over $1,000, "
                "a shared origin across accounts, or an undocumented pattern), and this case "
                "meets none of them.")

    with meter.stage("update_memory"):
        graph_case_id = ""
        written = False
        try:
            record = {
                "case_id": state.case_id, "graph_case_id": f"CASE-{state.case_id}",
                "customer_id": state.customer_id, "card_id": state.card_id,
                # card_key, not just card_id: the CASE_CARD edge keys on the card VERTEX, whose
                # primary id is the reconstructed key. Without it the edge was never written and
                # `recurring_entities_across_cases` - which traverses it to find cards appearing
                # in more than one of our own cases - had nothing to walk.
                "card_key": state.card_key,
                "verdict": s.verdict, "fraud_probability": round(s.probability, 3),
                "pattern": pattern.pattern, "exposure_usd": exposure,
                "affected_txn_ids": affected, "status": status_for(s.verdict, final),
                "actions": final.action_names, "summary": summary,
                "similar_prior_cases": state.similar_cases,
                "evidence": [e.__dict__ for e in state.evidence],
                "decisions": state.decisions,
                "opened_at": str(state.opened_at),
            }
            graph_case_id = meter.measure("graph.write_case")(be.write_case)(record)
            # `written_to_graph` must reflect a real write, so read it back rather than trust it
            written = meter.measure("graph.read_case")(be.read_case)(graph_case_id) is not None
            meter.event("memory", graph_case_id=graph_case_id, read_back=written)
        except Exception as exc:                            # noqa: BLE001 - reported, not fatal
            state.decide(f"Case memory write failed: {type(exc).__name__}: {exc}")

    meter.finish()

    # Diagnostics live BESIDE the answer, never inside it. The answer schema is fixed and
    # "nothing extra, nothing missing"; the rubric's contributions, the value-of-information
    # comparison and the decision log are what make the case explainable to a human, so they go
    # in a sidecar the UI and the write-up read. Adding them to the graded file would risk a
    # schema mismatch on the one artifact that must not have one.
    diagnostics = {
        "case_id": state.case_id,
        "assessment": s.to_json(),
        "pattern": pattern.to_json(),
        "recurrence": {"matches": bool(state.recurrence and state.recurrence.matches),
                       "claim": state.recurrence.as_evidence_claim() if state.recurrence else ""},
        "evidence_value": state.voi_log,
        "decisions": state.decisions,
        "trigger": {"type": state.trigger_type, "narrative": state.narrative,
                    "flagged_txn_id": state.flagged_txn_id,
                    "opened_at": str(state.opened_at)},
        "card": {"card_id": state.card_id, "card_key": state.card_key,
                 "n_cards": state.n_cards},
        "meter": meter.to_json(),
        "trace": meter.trace,
    }

    answer = Answer(
        case_id=state.case_id,
        case=Case(
            status=status_for(s.verdict, final),
            verdict=s.verdict,
            fraud_probability=round(s.probability, 3),
            pattern=pattern.pattern,
            pattern_description=pattern.description,
            affected_txn_ids=affected,
            first_suspicious_txn_id=(min(affected, key=lambda t: timestamps.get(t, ""))
                                     if affected else ""),
            connected_card_ids=state.connected_card_ids,
            connected_device_profiles=(state.connected_profiles
                                       if state.rubric_input.ring_signal else []),
            exposure_usd=exposure,
            evidence=state.evidence,
            similar_prior_cases=state.similar_cases,
            summary=summary,
            written_to_graph=written,
            graph_case_id=graph_case_id,
        ),
        evidence_requests=state.requests,
        initial_actions=initial,
        final_actions=final,
        what_changed=describe_change(initial, final, state),
        sar=sar,
        stop_reason=state.stop_reason,
        **Meter.answer_fields(meter),
    )
    return answer, meter, state, diagnostics


def describe_change(initial: Decision, final: Decision, state: CaseState) -> str:
    """What the evidence request changed, generated by diffing rather than narrated.

    The validator requires the literal string "nothing" when no request was made, and the brief
    asks what changed when one was. Deriving it from the two action sets means the field can
    never disagree with them.
    """
    if not state.requests:
        return "nothing"
    before = {r.action for r in initial.recommendations}
    after = {r.action for r in final.recommendations}
    added = sorted(a.value for a in after - before)
    removed = sorted(a.value for a in before - after)
    if not added and not removed:
        return (f"The {state.requests[-1].type} response did not change the recommended actions; "
                "it raised confidence in the ones already chosen.")
    parts = []
    if added:
        parts.append("added " + ", ".join(added))
    if removed:
        parts.append("withdrew " + ", ".join(removed))
    return (f"After the {', '.join(r.type for r in state.requests)} response, the recommendation "
            + " and ".join(parts) + ".")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Run every case in the pack, deterministically.")
    ap.add_argument("--data", default="data", help="directory of parquet built by features.build")
    ap.add_argument("--out", default="cases", help="where the answer files go")
    ap.add_argument("--only", default="", help="comma-separated case ids, for iterating on one")
    ap.add_argument("--backend", choices=("local", "tigergraph"), default="local",
                    help="tigergraph is the default in the demo; local is the fallback and the "
                         "backtest engine. The contract test proves they agree.")
    a = ap.parse_args()

    t0 = time.time()
    if not os.path.exists(os.path.join(a.data, "case_pack.parquet")):
        raise SystemExit("\n".join([
            "",
            f"No case pack in {a.data}/ - there is nothing to investigate.",
            "",
            "  for the real dataset:  python -m src.features.build",
            "  for the fixture:       python -m src.fixtures.generate",
            "                         then --data data_fixture",
            "",
        ]))
    if a.backend == "tigergraph":
        # No silent fallback. A run that quietly used parquet while reporting TigerGraph would be
        # worse than one that failed, because the claim in the write-up would be false.
        from ..graph.tigergraph import from_env
        be = from_env()
        print(f"backend: TigerGraph at {be.host} (graph {be.graph})")
    else:
        be = LocalBackend(a.data)
    pack = pd.read_parquet(os.path.join(a.data, "case_pack.parquet"))
    if a.only:
        wanted = {c.strip() for c in a.only.split(",")}
        pack = pack[pack.case_id.isin(wanted)]

    # Chronological order, so a later case can retrieve an earlier one from case memory - the
    # brief asks that the next investigation be able to find the last one, and order is the only
    # thing that makes that testable rather than aspirational.
    pack = pack.sort_values("opened_at")

    # Exposure must be recomputed from the dataset to the cent (validator invariant 4), and the
    # activity dates come from the same place. Both are read from parquet whichever backend is
    # driving the investigation: pulling 590,742 amounts back over HTTP to sum two of them would
    # be a strange way to use a graph.
    amounts, timestamps = load_ledger(a.data)

    os.makedirs(a.out, exist_ok=True)
    meters: list[Meter] = []
    print(f"{'case':<10} {'verdict':<11} {'p':>5} {'pattern':<28} {'exposure':>10} "
          f"{'asked':<20} actions")
    print("-" * 132)
    diagnostics = []
    for row in pack.itertuples():
        answer, meter, state, diag = investigate(be, row, amounts, timestamps)
        answer.write(a.out)
        meters.append(meter)
        diagnostics.append(diag)
        asked = ", ".join(r.type for r in answer.evidence_requests) or "-"
        print(f"{answer.case_id:<10} {answer.case.verdict:<11} "
              f"{answer.case.fraud_probability:>5.2f} {answer.case.pattern:<28} "
              f"{answer.case.exposure_usd:>10,.2f} {asked:<20} "
              f"{', '.join(r['action'] for r in answer.final_actions.to_json())}")

    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "backend": a.backend, "data_dir": os.path.abspath(a.data),
                "llm": None, "note": "deterministic run - no model calls",
                "run": roll_up(meters),
                "cases": [m.to_json() for m in meters]}
    with open(os.path.join(a.out, "_run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(a.out, "_diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump({d["case_id"]: d for d in diagnostics}, fh, indent=2, default=str)

    r = manifest["run"]
    print(f"\n{r['cases']} cases in {time.time() - t0:.1f}s | "
          f"{r['total_tool_calls']} graph queries | {r['total_llm_calls']} model calls | "
          f"${r['total_cost_usd']:.4f}")
    print(f"evidence from the graph: {r['mean_deterministic_evidence_share']:.0%} of all items")
    print(f"wrote {a.out}/*.json and {a.out}/_run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
