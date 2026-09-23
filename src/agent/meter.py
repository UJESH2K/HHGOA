"""Instrumentation: tool calls, tokens, latency, and what the run cost.

WHY THIS EXISTS BEFORE THE AGENT DOES. `tool_calls`, `tokens` and `latency_s` are required
fields in every answer file, and "missing fields score zero for that part". Retrofitting counters
onto a finished agent means threading a mutable object through seven nodes after the fact and
discovering that half the call sites never had access to it. So the meter is written first and
the nodes are built around it.

The second reason is the efficiency claim. The architecture's whole bet is that the deterministic
layer does the gathering and the model only judges and writes - which is either true and
measurable, or marketing. This module makes it measurable: cost per case, cache hit rate, tokens
by stage and by model, and the share of evidence that came from a graph query rather than from a
model. A number we measured beats an adjective we chose.

Prices are first-party Anthropic API rates per million tokens, current at the time of writing.
They live in one dict so that a rate change is a one-line edit rather than a hunt.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

# $/1M tokens: (input, output). Cache reads bill at a tenth of input; cache writes at 1.25x.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25
BATCH_DISCOUNT = 0.50
"""The Batch API halves the price of anything not latency-sensitive. The backtest sweeps qualify;
the graded run does not, because we want to watch it happen."""


def cost_usd(model: str, *, input_tokens: int = 0, output_tokens: int = 0,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0,
             batch: bool = False) -> float:
    """What one call cost. Unknown models cost 0 and are reported as unpriced rather than guessed."""
    if model not in PRICING:
        return 0.0
    in_rate, out_rate = PRICING[model]
    total = (input_tokens * in_rate
             + cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
             + cache_write_tokens * in_rate * CACHE_WRITE_MULTIPLIER
             + output_tokens * out_rate) / 1_000_000
    return total * (BATCH_DISCOUNT if batch else 1.0)


@dataclass
class LLMCall:
    model: str
    stage: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float = 0.0
    batch: bool = False

    @property
    def billable_input(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cost(self) -> float:
        return cost_usd(self.model, input_tokens=self.input_tokens,
                        output_tokens=self.output_tokens,
                        cache_read_tokens=self.cache_read_tokens,
                        cache_write_tokens=self.cache_write_tokens, batch=self.batch)


@dataclass
class ToolCall:
    name: str
    stage: str
    latency_s: float
    ok: bool = True
    error: str = ""


@dataclass
class Meter:
    """One meter per case. Threaded through the nodes; never global.

    A global counter would make the backtest's numbers meaningless the first time two cases were
    replayed concurrently, and concurrency is exactly how 200 replays become affordable.
    """
    case_id: str = ""
    started_at: float = field(default_factory=time.perf_counter)
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls_log: list[ToolCall] = field(default_factory=list)
    evidence_by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Ordered record of what the investigation did and when, in milliseconds from the start of
    # the case. The UI replays it; nothing in the answer depends on it.
    trace: list[dict] = field(default_factory=list)
    # Set by `finish()`. Without it `latency_s` kept counting after the case was done, so a run
    # manifest written at the end credited every case with the time spent on the cases after it.
    finished_at: float | None = None
    _stage: str = "init"
    _stage_times: dict[str, float] = field(default_factory=dict)

    # --- stages ----------------------------------------------------------------------------

    @contextmanager
    def stage(self, name: str):
        """Attribute everything measured inside the block to a named node.

        Per-stage attribution is what turns "the run cost $0.16" into "83% of the cost is the
        Explain node writing prose, so cache the policy corpus and shorten the summary" - which
        is the difference between reporting a number and acting on one.
        """
        previous, self._stage = self._stage, name
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self._stage = previous
            self._stage_times[name] = self._stage_times.get(name, 0.0) + (time.perf_counter() - t0)

    # --- recording -------------------------------------------------------------------------

    def record_llm(self, model: str, usage: Any, *, latency_s: float = 0.0,
                   batch: bool = False, stage: str | None = None) -> LLMCall:
        """Record one model call from an SDK `usage` object.

        Read by attribute with a zero default rather than by key, so a new usage field in a
        future SDK version cannot raise here, and a missing one cannot silently become a
        KeyError in the middle of a graded run. `cache_read_input_tokens` is the field the
        caching claim rests on - if it stays zero across a run, something in the prompt prefix is
        changing between cases and the cost table is wrong.
        """
        call = LLMCall(
            model=model, stage=stage or self._stage,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            latency_s=latency_s, batch=batch)
        self.llm_calls.append(call)
        return call

    def record_tool(self, name: str, latency_s: float, *, ok: bool = True, error: str = "",
                    stage: str | None = None) -> None:
        self.tool_calls_log.append(
            ToolCall(name=name, stage=stage or self._stage, latency_s=latency_s, ok=ok,
                     error=error))
        started = (time.perf_counter() - latency_s - self.started_at) * 1000
        self.trace.append({"t_ms": round(started, 2), "kind": "query", "name": name,
                           "stage": stage or self._stage, "ms": round(latency_s * 1000, 2),
                           "ok": ok, **({"error": error} if error else {})})

    def event(self, kind: str, **data: Any) -> None:
        """Append a step to the trace - an assessment, a request, a recommendation."""
        self.trace.append({"t_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
                           "kind": kind, "stage": self._stage, **data})

    def record_evidence(self, source: str) -> None:
        """Count an evidence item by where it came from: graph | document | customer | external.

        This is the denominator of the "the model does not gather, it judges" claim.
        """
        self.evidence_by_source[source] += 1

    def measure(self, name: str) -> Callable:
        """Decorator that counts and times a tool call.

        Wraps rather than replaces, so a failing graph query is still recorded as a call that
        happened - an agent that made nine queries and lost three is not an agent that made six.
        """
        def outer(fn: Callable) -> Callable:
            @wraps(fn)
            def inner(*a, **kw):
                t0 = time.perf_counter()
                try:
                    out = fn(*a, **kw)
                except Exception as exc:                      # noqa: BLE001 - recorded, re-raised
                    self.record_tool(name, time.perf_counter() - t0, ok=False,
                                     error=f"{type(exc).__name__}: {exc}")
                    raise
                self.record_tool(name, time.perf_counter() - t0)
                return out
            return inner
        return outer

    # --- totals ----------------------------------------------------------------------------

    @property
    def tool_calls(self) -> int:
        return len(self.tool_calls_log)

    @property
    def tokens(self) -> int:
        """Total tokens, input plus output, cache reads included.

        The brief asks for `tokens` without defining it. Counting cache reads is the honest
        choice: they are tokens the model processed, they are billed, and excluding them would
        let the caching strategy flatter the number it is supposed to be measured by. The cost
        figure prices them correctly at a tenth of the input rate, so nothing is double-counted.
        """
        return sum(c.billable_input + c.output_tokens for c in self.llm_calls)

    @property
    def latency_s(self) -> float:
        return (self.finished_at or time.perf_counter()) - self.started_at

    def finish(self) -> None:
        if self.finished_at is None:
            self.finished_at = time.perf_counter()

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self.llm_calls)

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache. Zero across a run means a broken prefix."""
        total_in = sum(c.billable_input for c in self.llm_calls)
        if not total_in:
            return 0.0
        return sum(c.cache_read_tokens for c in self.llm_calls) / total_in

    @property
    def deterministic_evidence_share(self) -> float:
        """Share of evidence items that came from the graph rather than from a model or a person."""
        total = sum(self.evidence_by_source.values())
        if not total:
            return 0.0
        return self.evidence_by_source.get("graph", 0) / total

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for t in self.tool_calls_log if not t.ok)

    # --- output ----------------------------------------------------------------------------

    def answer_fields(self) -> dict:
        """Exactly the three instrumentation fields the answer file requires."""
        return {"tool_calls": self.tool_calls, "tokens": self.tokens,
                "latency_s": round(self.latency_s, 2)}

    def to_json(self) -> dict:
        """The full ledger. Goes in the run manifest, not in the answer file.

        The answer schema is fixed and "nothing extra, nothing missing" - so the cost detail
        lives beside the answers in `cases/_run_manifest.json`, where the UI and the blog post
        can read it without risking a schema mismatch on a graded file.
        """
        by_stage: dict[str, dict[str, float]] = {}
        for c in self.llm_calls:
            row = by_stage.setdefault(c.stage, {"llm_calls": 0, "tokens": 0, "cost_usd": 0.0})
            row["llm_calls"] += 1
            row["tokens"] += c.billable_input + c.output_tokens
            row["cost_usd"] = round(row["cost_usd"] + c.cost, 6)
        for t in self.tool_calls_log:
            row = by_stage.setdefault(t.stage, {"llm_calls": 0, "tokens": 0, "cost_usd": 0.0})
            row["tool_calls"] = row.get("tool_calls", 0) + 1
            row["tool_seconds"] = round(row.get("tool_seconds", 0.0) + t.latency_s, 3)
        for name, seconds in self._stage_times.items():
            by_stage.setdefault(name, {"llm_calls": 0, "tokens": 0, "cost_usd": 0.0})
            by_stage[name]["wall_seconds"] = round(seconds, 3)

        by_model: dict[str, dict[str, float]] = {}
        for c in self.llm_calls:
            row = by_model.setdefault(c.model, {"calls": 0, "input": 0, "cache_read": 0,
                                                "output": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["input"] += c.input_tokens
            row["cache_read"] += c.cache_read_tokens
            row["output"] += c.output_tokens
            row["cost_usd"] = round(row["cost_usd"] + c.cost, 6)

        return {
            "case_id": self.case_id,
            **self.answer_fields(),
            "llm_calls": len(self.llm_calls),
            "cost_usd": round(self.cost, 6),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "failed_tool_calls": self.failed_tool_calls,
            "evidence_by_source": dict(self.evidence_by_source),
            "deterministic_evidence_share": round(self.deterministic_evidence_share, 4),
            "unpriced_models": sorted({c.model for c in self.llm_calls
                                       if c.model not in PRICING}),
            "by_stage": by_stage,
            "by_model": by_model,
            "tools": sorted({t.name for t in self.tool_calls_log}),
        }

    def summary_line(self) -> str:
        return (f"{self.case_id or 'case'}: {self.tool_calls} tool calls, "
                f"{len(self.llm_calls)} llm calls, {self.tokens:,} tokens, "
                f"${self.cost:.4f}, {self.latency_s:.1f}s, "
                f"cache {self.cache_hit_rate:.0%}")


def roll_up(meters: list[Meter]) -> dict:
    """Run-level totals across cases - the table that goes in the blog post."""
    if not meters:
        return {"cases": 0}
    return {
        "cases": len(meters),
        "total_cost_usd": round(sum(m.cost for m in meters), 4),
        "mean_cost_usd": round(sum(m.cost for m in meters) / len(meters), 4),
        "max_cost_usd": round(max(m.cost for m in meters), 4),
        "total_tokens": sum(m.tokens for m in meters),
        "total_tool_calls": sum(m.tool_calls for m in meters),
        "total_llm_calls": sum(len(m.llm_calls) for m in meters),
        "total_latency_s": round(sum(m.latency_s for m in meters), 1),
        "mean_latency_s": round(sum(m.latency_s for m in meters) / len(meters), 2),
        "mean_cache_hit_rate": round(sum(m.cache_hit_rate for m in meters) / len(meters), 4),
        "mean_deterministic_evidence_share": round(
            sum(m.deterministic_evidence_share for m in meters) / len(meters), 4),
        "failed_tool_calls": sum(m.failed_tool_calls for m in meters),
    }
