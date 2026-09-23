"""The analyst copilot: Claude answers questions about a finished case, and never decides it.

WHERE THE MODEL SITS, AND WHY THERE. The investigation is deterministic on purpose - the verdict,
the probability, the actions and their approval routes come from the rubric and the policy engine,
so every answer file can be reproduced to the byte and audited line by line. What a deterministic
engine cannot do is talk: an L1 analyst approving a BLOCK_CARD wants to ask "why not just verify
first?", and an L2 approver wants the case in three sentences before signing a report. That is the
job the model is good at and the engine is not, so that is the job it gets.

The model reads; it does not write. It sees the answer file, the diagnostics (every rubric
contribution, every evidence request that was priced, the decision log) and the Fraud Policy, and
it is told that it may explain the decision and may disagree with it out loud, but that changing
it is a human's job through the approval route. Nothing it says is written back to the case.

COST. The policy text and the case file are a stable prefix, marked for prompt caching, so a
follow-up question on the same case re-reads them at a tenth of the input price. Each call is
metered with the same `Meter` the investigation uses, and the page shows tokens, cache hits,
latency and dollars under every answer - the cost claim is made per call, in front of the reader.

    python -m src.agent.copilot HHG-014 "Why was step-up chosen over calling the customer?"
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

from .meter import cost_usd

MODEL = os.environ.get("COPILOT_MODEL", "claude-opus-5")
# Low effort: these are reading-comprehension answers over a document already in context, and
# the page is waiting on the first token. Raise it for the report drafts if they need more care.
EFFORT = os.environ.get("COPILOT_EFFORT", "low")
MAX_TOKENS = 4000

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(HERE, "policy_corpus.md")

SYSTEM = """You are the analyst copilot inside a fraud-investigation console built on TigerGraph.

An automated investigation has already worked this case. Its verdict, fraud probability, pattern,
recommended actions and approval routes were produced by a deterministic rubric and a policy
engine, from graph queries over the bank's transaction graph. You did not make those decisions
and you cannot change them. Your job is to help a human analyst understand, check and act on them.

How to answer:
- Latency-sensitive; begin your visible answer immediately.
- Use only the case file, the diagnostics and the Fraud Policy below. If the answer is not in
  them, say so plainly rather than guessing.
- Cite evidence by its index in the case file's evidence list, like [E3], and cite policy rules
  by their identifiers, like R1 or section 3a.
- Anything marked SIMULATED is an assumed response, not collected evidence. Say so when you rely
  on it.
- You may disagree with the automated decision if the evidence supports that. Say what you
  would do instead and which approval route that change would need; the analyst decides.
- Be brief: a short paragraph or a few bullets. No preamble, no restating the question.

The Fraud Policy and the five known patterns follow.

"""


def _policy() -> str:
    with open(POLICY_PATH, encoding="utf-8") as fh:
        return fh.read()


def case_context(answer: dict, diagnostics: dict | None) -> str:
    """The case as the model sees it: evidence indexed, diagnostics trimmed to what explains.

    The trace and the per-stage meter are left out - they describe how fast the engine ran, not
    why it decided what it did, and they would cost tokens on every call.
    """
    d = dict(diagnostics or {})
    for noise in ("trace", "meter"):
        d.pop(noise, None)
    case = answer.get("case", {})
    evidence = "\n".join(f"[E{i + 1}] ({e.get('source')}, {e.get('ref')}) {e.get('claim')}"
                         for i, e in enumerate(case.get("evidence", [])))
    slim = {k: v for k, v in answer.items() if k != "case"}
    slim["case"] = {k: v for k, v in case.items() if k != "evidence"}
    return (f"<case_file id=\"{answer.get('case_id')}\">\n"
            f"{json.dumps(slim, indent=1, ensure_ascii=False)}\n</case_file>\n\n"
            f"<evidence>\n{evidence}\n</evidence>\n\n"
            f"<diagnostics>\n{json.dumps(d, indent=1, ensure_ascii=False, default=str)}\n"
            f"</diagnostics>")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def ask(answer: dict, diagnostics: dict | None, question: str,
        history: list[dict] | None = None) -> Iterator[dict]:
    """Stream an answer. Yields {"type": "text", "text": ...} deltas, then one "done" event.

    `history` is prior turns on this same case as {"role", "content"} pairs, so a follow-up can
    say "and the other card?". It sits after the cached prefix, so it never invalidates it.
    """
    import anthropic

    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": [
            # One breakpoint after the case context caches policy + case together; the policy
            # alone is under the minimum cacheable prefix, the two together are not.
            {"type": "text", "text": case_context(answer, diagnostics),
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "I will ask questions about this case."},
        ],
    }, {"role": "assistant", "content": "Understood. Ask away."}]
    for turn in (history or [])[-8:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
    messages.append({"role": "user", "content": question})

    t0 = time.perf_counter()
    first_token_s = None
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM + _policy()}],
        messages=messages,
        output_config={"effort": EFFORT},
        # If a safety classifier declines, re-run on Anthropic's recommended fallback model
        # instead of returning a refusal to an analyst mid-investigation.
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
    ) as stream:
        for text in stream.text_stream:
            if first_token_s is None:
                first_token_s = time.perf_counter() - t0
            yield {"type": "text", "text": text}
        final = stream.get_final_message()

    u = final.usage
    usage = {
        "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
        "cache_read_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(u, "cache_creation_input_tokens", 0) or 0),
    }
    yield {
        "type": "done",
        "model": getattr(final, "model", MODEL),
        "stop_reason": final.stop_reason,
        "latency_s": round(time.perf_counter() - t0, 2),
        "first_token_s": round(first_token_s, 2) if first_token_s is not None else None,
        "usage": usage,
        "cost_usd": round(cost_usd(MODEL, **usage), 5),
    }


def main() -> int:
    import sys

    from ..config import load_env
    load_env()
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    case_id, question = sys.argv[1], " ".join(sys.argv[2:])
    with open(os.path.join("cases", f"{case_id}.json"), encoding="utf-8") as fh:
        answer = json.load(fh)
    with open(os.path.join("cases", "_diagnostics.json"), encoding="utf-8") as fh:
        diag = json.load(fh).get(case_id)
    for event in ask(answer, diag, question):
        if event["type"] == "text":
            print(event["text"], end="", flush=True)
        else:
            print("\n\n" + json.dumps({k: v for k, v in event.items() if k != "type"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
