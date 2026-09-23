"""The single place an evidence-request response is invented, and the place it is disclosed.

The organizers do not supply responses. An agent that asks the cardholder a question therefore
has to assume an answer, and the answer it assumes determines the action it recommends - which
makes this the most dangerous module in the repo. Two rules follow, and they are the whole design:

  1. **One place.** If the model were allowed to imagine a reply inline, the case file would
     contain a recommendation justified by a conversation that never happened, and in the demo
     that is indistinguishable from fabrication. Every simulated response comes from here.

  2. **Say so, in the answer file.** `evidence_requests[].assumed_response` is a required field
     precisely because the organizers know the response is invented. This module produces the
     verbatim sentence that goes in it, including the reason the assumption was made and what
     would have followed from the other answers.

HOW THE ANSWER IS CHOSEN. The most likely branch under the same response model the
value-of-information selector used to decide the question was worth asking. That is deliberate:
using one model for "what might they say" and another for "what did they say" would let the agent
ask a question because a denial was plausible and then answer it with a confession.

It is also the conservative choice in the direction that matters. A customer-report trigger means
the cardholder has already disputed the charge, so we never simulate them taking it back; and on
a case the evidence says is dull, the model's most likely answer is that the charge is theirs -
which closes the case rather than blocking a card on an assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..scoring.rubric import RubricInput
from .voi import (ANALYST_INFO, CUSTOMER_VALIDATION, STEP_UP_AUTH, Option, apply_response)

# What each response literally means, for the case file.
MEANING = {
    ("customer_validation", "denied"): "the cardholder states they did not make the transaction",
    ("customer_validation", "confirmed"): "the cardholder states the transaction is theirs",
    ("customer_validation", "no_reply"): "the cardholder does not respond within the policy's "
                                         "24-hour window",
    ("step_up_auth", "passed"): "the cardholder completes the step-up challenge",
    ("step_up_auth", "failed"): "the step-up challenge is not completed",
    ("analyst_info", "confirms_link"): "the analyst confirms the shared-origin link",
    ("analyst_info", "confirms_known_relationship"): "the analyst confirms an established "
                                                     "merchant relationship on this account",
    ("analyst_info", "nothing_new"): "the analyst has nothing to add beyond what the graph shows",
}


@dataclass
class SimulatedResponse:
    request_type: str
    response: str
    probability: float
    statement: str
    """The verbatim sentence for `evidence_requests[].assumed_response`."""
    counterfactual: str
    """What the other answers would have led to - so the reader can weigh the assumption."""

    def apply(self, i: RubricInput) -> RubricInput:
        return apply_response(i, self.request_type, self.response)


def respond(option: Option) -> SimulatedResponse:
    """Simulate the response to a request the selector decided was worth making.

    Takes the whole `Option` rather than a request type, because the option already carries the
    branch probabilities and the consequence of each branch - which is exactly the material the
    disclosure sentence needs. Nothing new is invented here; a branch is chosen and described.
    """
    if not option.branches:
        raise ValueError(f"{option.request_type} has no branches to choose from")

    chosen = max(option.branches, key=lambda b: b.probability)
    meaning = MEANING.get((option.request_type, chosen.response), chosen.response)

    statement = (
        f"SIMULATED (no response data is provided with this dataset): {meaning}. "
        f"Assumed because it is the most likely answer at this point in the investigation "
        f"(p={chosen.probability:.2f} under the response model in src/agent/voi.py), given a "
        f"fraud probability of {chosen.probability_after:.2f} before the request.")

    others = [b for b in option.branches if b is not chosen]
    if others:
        counterfactual = "Had the answer been " + "; or ".join(
            f"{MEANING.get((option.request_type, b.response), b.response)} (p={b.probability:.2f}), "
            f"the recommendation would have moved to {b.posture_after} with a fraud probability "
            f"of {b.probability_after:.2f}"
            for b in others) + "."
    else:
        counterfactual = "No alternative answer was modelled for this request type."

    return SimulatedResponse(request_type=option.request_type, response=chosen.response,
                             probability=chosen.probability, statement=statement,
                             counterfactual=counterfactual)


def is_simulated_everywhere(requests: list[dict]) -> bool:
    """Every assumed response must announce itself. Used by the end-to-end test.

    A case file that presents an invented reply as though it were collected evidence is worse
    than one that asks no questions at all.
    """
    return all("SIMULATED" in (r.get("assumed_response") or "") for r in requests)


__all__ = ["SimulatedResponse", "respond", "is_simulated_everywhere", "MEANING",
           "CUSTOMER_VALIDATION", "STEP_UP_AUTH", "ANALYST_INFO"]
