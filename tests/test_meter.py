"""Meter tests. No dataset, no credentials, no API calls.

`tool_calls`, `tokens` and `latency_s` are required answer-file fields, and the cost ledger is
the evidence behind the efficiency claim in the write-up. Both are worth a test: a counter that
silently reads zero is worse than no counter, because it looks like an answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.agent.meter import (BATCH_DISCOUNT, CACHE_READ_MULTIPLIER, PRICING, Meter, cost_usd,
                             roll_up)

OPUS, SONNET, HAIKU = "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"


@dataclass
class Usage:
    """The shape the Anthropic SDK returns on `response.usage`."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


# --- pricing --------------------------------------------------------------------------------

def test_a_priced_call_costs_what_the_rate_card_says():
    # 1M input + 1M output on Opus 5 at $5 / $25
    assert cost_usd(OPUS, input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(30.0)


def test_cache_reads_are_a_tenth_of_input():
    full = cost_usd(OPUS, input_tokens=100_000)
    cached = cost_usd(OPUS, cache_read_tokens=100_000)
    assert cached == pytest.approx(full * CACHE_READ_MULTIPLIER)


def test_cache_writes_cost_a_premium():
    assert cost_usd(OPUS, cache_write_tokens=100_000) > cost_usd(OPUS, input_tokens=100_000)


def test_batch_halves_the_bill():
    a = cost_usd(HAIKU, input_tokens=500_000, output_tokens=50_000)
    b = cost_usd(HAIKU, input_tokens=500_000, output_tokens=50_000, batch=True)
    assert b == pytest.approx(a * BATCH_DISCOUNT)


def test_model_tiers_are_ordered_as_expected():
    same = dict(input_tokens=100_000, output_tokens=10_000)
    assert cost_usd(HAIKU, **same) < cost_usd(SONNET, **same) < cost_usd(OPUS, **same)


def test_an_unknown_model_costs_zero_and_is_reported_rather_than_guessed():
    m = Meter(case_id="HHG-001")
    m.record_llm("some-model-we-have-not-priced", Usage(input_tokens=1000, output_tokens=100))
    assert m.cost == 0.0
    assert m.to_json()["unpriced_models"] == ["some-model-we-have-not-priced"]


def test_every_priced_model_has_input_cheaper_than_output():
    for model, (inp, out) in PRICING.items():
        assert 0 < inp < out, model


# --- the three required fields --------------------------------------------------------------

def test_tokens_counts_input_output_and_cache_reads():
    m = Meter()
    m.record_llm(OPUS, Usage(input_tokens=1_000, output_tokens=500,
                             cache_read_input_tokens=9_000))
    assert m.tokens == 10_500


def test_tool_calls_counts_failures_too():
    """An agent that made nine queries and lost three did not make six."""
    m = Meter()

    @m.measure("graph.card_history")
    def ok():
        return 1

    @m.measure("graph.ring_signals")
    def boom():
        raise RuntimeError("workspace asleep")

    ok()
    ok()
    with pytest.raises(RuntimeError):
        boom()
    assert m.tool_calls == 3
    assert m.failed_tool_calls == 1
    assert "workspace asleep" in m.tool_calls_log[-1].error


def test_answer_fields_are_exactly_the_three_required_keys():
    m = Meter()
    m.record_llm(SONNET, Usage(input_tokens=10, output_tokens=5))
    assert set(m.answer_fields()) == {"tool_calls", "tokens", "latency_s"}
    assert isinstance(m.answer_fields()["tool_calls"], int)
    assert isinstance(m.answer_fields()["tokens"], int)
    assert m.answer_fields()["latency_s"] >= 0


def test_latency_is_monotonic():
    m = Meter()
    first = m.latency_s
    assert m.latency_s >= first >= 0


# --- attribution ----------------------------------------------------------------------------

def test_stages_attribute_calls_to_the_right_node():
    m = Meter(case_id="HHG-006")
    with m.stage("investigate"):
        m.record_llm(HAIKU, Usage(input_tokens=100, output_tokens=10))

        @m.measure("graph.prior_cases")
        def q():
            return []
        q()
    with m.stage("explain"):
        m.record_llm(OPUS, Usage(input_tokens=200, output_tokens=900))

    j = m.to_json()
    assert j["by_stage"]["investigate"]["llm_calls"] == 1
    assert j["by_stage"]["investigate"]["tool_calls"] == 1
    assert j["by_stage"]["explain"]["llm_calls"] == 1
    assert j["by_stage"]["explain"]["cost_usd"] > j["by_stage"]["investigate"]["cost_usd"]


def test_nested_stages_restore_the_outer_one():
    m = Meter()
    with m.stage("outer"):
        with m.stage("inner"):
            m.record_llm(HAIKU, Usage(input_tokens=1, output_tokens=1))
        m.record_llm(HAIKU, Usage(input_tokens=1, output_tokens=1))
    stages = [c.stage for c in m.llm_calls]
    assert stages == ["inner", "outer"]


def test_stage_can_be_overridden_per_call():
    m = Meter()
    m.record_llm(HAIKU, Usage(input_tokens=1), stage="assess")
    assert m.llm_calls[0].stage == "assess"


def test_by_model_breaks_the_bill_down():
    m = Meter()
    m.record_llm(HAIKU, Usage(input_tokens=1000, output_tokens=100))
    m.record_llm(HAIKU, Usage(input_tokens=1000, output_tokens=100))
    m.record_llm(OPUS, Usage(input_tokens=1000, output_tokens=1000))
    by_model = m.to_json()["by_model"]
    assert by_model[HAIKU]["calls"] == 2
    assert by_model[OPUS]["calls"] == 1
    assert sum(r["cost_usd"] for r in by_model.values()) == pytest.approx(m.cost, abs=1e-6)


# --- the claims the ledger has to support ---------------------------------------------------

def test_cache_hit_rate_is_measured_not_asserted():
    m = Meter()
    m.record_llm(OPUS, Usage(input_tokens=2_000, cache_read_input_tokens=8_000,
                             output_tokens=500))
    assert m.cache_hit_rate == pytest.approx(0.8)


def test_cache_hit_rate_is_zero_when_nothing_was_cached():
    """If this stays zero across a run, the prompt prefix is changing between cases."""
    m = Meter()
    m.record_llm(OPUS, Usage(input_tokens=10_000, output_tokens=500))
    assert m.cache_hit_rate == 0.0


def test_cache_hit_rate_of_an_empty_meter_is_zero_not_an_error():
    assert Meter().cache_hit_rate == 0.0
    assert Meter().deterministic_evidence_share == 0.0
    assert Meter().tokens == 0


def test_deterministic_evidence_share_is_countable():
    m = Meter()
    for _ in range(17):
        m.record_evidence("graph")
    m.record_evidence("document")
    m.record_evidence("customer")
    assert m.deterministic_evidence_share == pytest.approx(17 / 19)
    assert m.to_json()["evidence_by_source"]["graph"] == 17


# --- run level ------------------------------------------------------------------------------

def test_roll_up_of_no_cases_is_not_a_crash():
    assert roll_up([]) == {"cases": 0}


def test_roll_up_totals_the_run():
    meters = []
    for n in range(20):
        m = Meter(case_id=f"HHG-{n + 1:03d}")
        m.record_llm(OPUS, Usage(input_tokens=2_000, cache_read_input_tokens=8_000,
                                 output_tokens=1_500))
        m.record_evidence("graph")
        meters.append(m)
    r = roll_up(meters)
    assert r["cases"] == 20
    assert r["total_llm_calls"] == 20
    assert r["mean_cost_usd"] == pytest.approx(r["total_cost_usd"] / 20)
    assert r["mean_cache_hit_rate"] == pytest.approx(0.8)
    # the budget claim in STRATEGY.md section 7: an Opus 5 run of 20 cases stays in single digits
    assert r["total_cost_usd"] < 10.0


def test_summary_line_mentions_the_numbers_that_matter():
    m = Meter(case_id="HHG-006")
    m.record_llm(OPUS, Usage(input_tokens=1_000, output_tokens=200))
    line = m.summary_line()
    for fragment in ("HHG-006", "tool calls", "tokens", "$", "cache"):
        assert fragment in line
