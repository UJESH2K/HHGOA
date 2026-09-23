"""The whole chain, on synthetic data with planted ground truth.

    fixtures/*.csv -> features.build -> LocalBackend -> agent.run -> cases/*.json -> validate

This is the test that the two policy bugs would have failed. Unit tests cannot catch "the agent
recommends opening a case and nothing else on every customer report", because nothing is wrong
with any individual function - the gap is between them. So this runs all twenty cases and checks
the output the judges will read.

It is slow by the standards of this suite (a few seconds) and it earns it. Marked `slow` so the
fast suite stays fast: `pytest -m "not slow"` skips it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from src.fixtures.generate import CASES, EXPECTED_PATTERN, build_fixture, expected_verdict, write
from src.fixtures.generate import SEED
from src.io.validate import Dataset, validate_answer, validate_set
from src.agent.simulate import is_simulated_everywhere

pytestmark = pytest.mark.slow

SHAPE = {case_id: shape for case_id, shape, _ in CASES}
TRIGGER = {case_id: trigger for case_id, _, trigger in CASES}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Generate fixtures, build parquet, investigate all 20, return the answers."""
    root = tmp_path_factory.mktemp("e2e")
    csv_dir, data_dir, out_dir = (str(root / "fixtures"), str(root / "data"),
                                  str(root / "cases"))

    write(build_fixture(SEED), csv_dir)

    from src.features.build import run as build_run
    build_run(csv_dir, data_dir)

    # Run the investigation as a subprocess, the way it will actually be invoked, so that an
    # import-time or CLI-level break is caught here rather than in front of the judges.
    proc = subprocess.run(
        [sys.executable, "-m", "src.agent.run", "--data", data_dir, "--out", out_dir],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert proc.returncode == 0, proc.stderr[-3000:]

    answers = {}
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".json") and not name.startswith("_"):
            with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
                obj = json.load(fh)
            answers[obj["case_id"]] = obj
    with open(os.path.join(out_dir, "_run_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    return {"answers": answers, "manifest": manifest, "data": data_dir, "out": out_dir,
            "stdout": proc.stdout}


# --- the deliverable ------------------------------------------------------------------------

def test_every_case_produces_an_answer_file(run):
    assert set(run["answers"]) == {c for c, _, _ in CASES}


def test_all_twelve_invariants_pass(run):
    """The submission gate. Anything here is a scoring error, not a style opinion."""
    ds = Dataset(run["data"])
    findings = []
    for answer in run["answers"].values():
        findings.extend(validate_answer(answer, ds))
    findings.extend(validate_set(run["answers"], ds))
    errors = [f for f in findings if f.level == "ERROR"]
    assert not errors, "\n".join(str(f) for f in errors)


# --- did it reach the right conclusions? ----------------------------------------------------

@pytest.mark.parametrize("case_id", [c for c, _, _ in CASES])
def test_verdict_matches_the_planted_shape(run, case_id):
    want = expected_verdict(SHAPE[case_id], TRIGGER[case_id])
    if want is None:
        pytest.skip("deliberately ambiguous - see test_ambiguous_cases_are_never_decided_blind")
    got = run["answers"][case_id]["case"]["verdict"]
    assert got == want, (f"{case_id} is a {SHAPE[case_id]} case arriving by "
                         f"{TRIGGER[case_id]}: expected {want}, got {got}")


def test_ambiguous_cases_are_never_decided_blind(run):
    """The restraint that 25% of the score rests on.

    A case built to be genuinely ambiguous may end up at any of the three verdicts - what it may
    not do is reach a confident one on the transaction alone. A confident verdict needs one of
    three things behind it:

      - a question the agent asked and got an answer to, or
      - the cardholder's own statement, already on the record because they raised the alert
        (asking someone to re-confirm a dispute they just filed is delay, not evidence), or
      - no confident verdict at all, left uncertain for a human.

    The middle case is the one that caught this test out first: HHG-011 reaches `fraud` without
    asking anything, and it is right to. The cardholder has already said the transaction is not
    theirs, R2 applies, and there is no third party left to ask.
    """
    for case_id in [c for c, sh, _ in CASES if sh == "ambiguous"]:
        answer = run["answers"][case_id]
        asked = bool(answer["evidence_requests"])
        disputed = TRIGGER[case_id] == "customer_report"
        verdict = answer["case"]["verdict"]
        assert asked or disputed or verdict == "uncertain", (
            f"{case_id} reached {verdict} on an ambiguous transaction with no question asked "
            "and no cardholder statement on the record")
        if disputed and not asked:
            # the statement has to actually be cited, not merely assumed from the trigger
            sources = {e["source"] for e in answer["case"]["evidence"]}
            assert "customer" in sources, f"{case_id} leans on a dispute it never recorded"


@pytest.mark.parametrize("shape,pattern", sorted(EXPECTED_PATTERN.items()))
def test_planted_patterns_are_identified(run, shape, pattern):
    ids = [c for c, sh, _ in CASES if sh == shape]
    for case_id in ids:
        assert run["answers"][case_id]["case"]["pattern"] == pattern, case_id


def test_the_card_testing_case_recommends_the_r5_actions(run):
    case_id = next(c for c, sh, _ in CASES if sh == "testing")
    actions = [a["action"] for a in run["answers"][case_id]["next_best_actions"]["final"]]
    assert "DECLINE_TRANSACTION" in actions
    assert "STEP_UP_AUTH" in actions


def test_the_ring_case_pulls_in_the_connected_cards_and_files(run):
    """R6: several cards showing fraud from a shared origin. Expensive to get wrong, so it is
    the one place the agent is allowed to reach for a regulatory filing on a modest exposure."""
    case_id = next(c for c, sh, _ in CASES if sh == "ring")
    answer = run["answers"][case_id]
    actions = [a["action"] for a in answer["next_best_actions"]["final"]]
    assert "MONITOR_CONNECTED_CARDS" in actions
    assert "FILE_REPORT" in actions
    assert answer["sar"]["file"] is True
    assert answer["case"]["connected_card_ids"]
    assert answer["case"]["connected_device_profiles"]


def test_subscription_disputes_are_not_blocked(run):
    """R7, and the whole reason the recurring-charge detector exists.

    R7's words are "recommend CREATE_CASE, VERIFY_WITH_CUSTOMER, and WARN_CUSTOMER. Do not
    block." It prescribes a restraint, not a verdict - so this asserts the restraint. Whether the
    case lands `legitimate` or `uncertain` is a calibration question; seizing the card of a
    customer disputing their own subscription is a policy breach either way.
    """
    for case_id in [c for c, sh, _ in CASES if sh == "subscription"]:
        answer = run["answers"][case_id]
        actions = [a["action"] for a in answer["next_best_actions"]["final"]]
        for seizure in ("BLOCK_CARD", "BLOCK_ALL_CARDS", "DECLINE_TRANSACTION"):
            assert seizure not in actions, f"{case_id} recommends {seizure} on a subscription"
        assert "WARN_CUSTOMER" in actions, f"{case_id} should warn the cardholder (R7)"
        assert answer["case"]["verdict"] in ("legitimate", "uncertain"), case_id


def test_a_legitimate_verdict_never_seizes_the_card(run):
    for case_id, answer in run["answers"].items():
        if answer["case"]["verdict"] == "legitimate":
            actions = [a["action"] for a in answer["next_best_actions"]["final"]]
            for seizure in ("BLOCK_CARD", "BLOCK_ALL_CARDS", "DECLINE_TRANSACTION"):
                assert seizure not in actions, f"{case_id} found no fraud but recommends {seizure}"


# --- calibration and restraint --------------------------------------------------------------

def test_the_verdict_split_is_not_unanimous(run):
    verdicts = [a["case"]["verdict"] for a in run["answers"].values()]
    assert len(set(verdicts)) > 1, verdicts
    assert verdicts.count("fraud") < len(verdicts)


def test_reports_are_filed_sparingly(run):
    """Closed-case history files on 7.1% of investigations. A third of the pack is miscalibrated."""
    filed = [c for c, a in run["answers"].items() if a["sar"]["file"]]
    assert len(filed) / len(run["answers"]) <= 0.33, filed


def test_probabilities_are_spread_not_saturated(run):
    probs = [a["case"]["fraud_probability"] for a in run["answers"].values()]
    assert min(probs) < 0.3 and max(probs) > 0.7, probs
    assert all(0.0 <= p <= 1.0 for p in probs)


# --- honesty ---------------------------------------------------------------------------------

def test_every_simulated_response_announces_itself(run):
    """A case file presenting an invented reply as collected evidence is worse than no reply."""
    for case_id, answer in run["answers"].items():
        assert is_simulated_everywhere(answer["evidence_requests"]), case_id


def test_cases_that_asked_a_question_record_what_changed(run):
    for case_id, answer in run["answers"].items():
        nba = answer["next_best_actions"]
        if answer["evidence_requests"]:
            assert nba["what_changed"] != "nothing", case_id
        else:
            assert nba["what_changed"] == "nothing", case_id
            assert nba["initial"] == nba["final"], case_id


def test_at_least_one_case_changed_its_recommendation_after_evidence(run):
    """The demo's central claim. If no case ever moves, the evidence loop is decoration."""
    moved = [c for c, a in run["answers"].items()
             if a["evidence_requests"]
             and a["next_best_actions"]["initial"] != a["next_best_actions"]["final"]]
    assert moved, "no case changed its recommendation after requesting evidence"


def test_every_case_is_written_to_the_graph_and_read_back(run):
    for case_id, answer in run["answers"].items():
        assert answer["case"]["written_to_graph"] is True, case_id
        assert answer["case"]["graph_case_id"], case_id


def test_evidence_is_gathered_not_narrated(run):
    """The efficiency claim: the deterministic layer gathers, the model only judges."""
    share = run["manifest"]["run"]["mean_deterministic_evidence_share"]
    assert share >= 0.6, share


def test_instrumentation_is_populated(run):
    for case_id, answer in run["answers"].items():
        assert answer["tool_calls"] > 0, case_id
        assert answer["latency_s"] >= 0
        assert answer["tokens"] == 0, "the deterministic run makes no model calls"


def test_the_run_manifest_carries_the_cost_ledger(run):
    r = run["manifest"]["run"]
    for key in ("cases", "total_cost_usd", "total_tool_calls", "mean_latency_s",
                "mean_deterministic_evidence_share"):
        assert key in r, key
    assert r["cases"] == len(CASES)


def test_every_case_explains_why_it_stopped(run):
    for case_id, answer in run["answers"].items():
        assert len(answer["stop_reason"]) > 40, case_id


def test_summaries_are_short_enough_to_read(run):
    for case_id, answer in run["answers"].items():
        summary = answer["case"]["summary"]
        assert summary
        assert summary.count(". ") <= 8, f"{case_id} summary is longer than the brief asks"
