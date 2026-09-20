# PLAN.md — Build Plan

> What's next, and what's done. Check items off as you go; add dated notes to the log at the
> bottom when a phase's plan changes. Re-read PROJECT.md and ARCHITECTURE.md before resuming
> after any gap.

**Deadline: not recorded — fill it in here on the next pass.** Estimates below are in focused
hours, not calendar time, so they survive whatever the real date turns out to be.

---

## Critical path

Everything that must happen in order, with nothing else on it:

```
P0 de-risk (TigerGraph reachable, MCP responds)   ──┐
P0 data prep (card_key, NEXT chain, baselines)    ──┼──► P2 agent loop ──► P3 exam run ──► P5 submit
P1 schema + load                                  ──┘         ▲
                                                              │
                                              P2.5 backtest ──┘  (tells you if P2 is any good)
```

P4 (UI) and the optional monitoring track hang off P3 and are cut first if time runs short.
**The one non-negotiable: 20 valid JSON files.** Everything else is worth less than that.

| Phase | Est. | Gate to exit |
|---|---|---|
| P0 de-risk | 2h | A GSQL query returns a row through MCP |
| P0 data prep | 4h | Sanity counts match PROJECT.md §3 exactly |
| P1 schema + load | 6h | All vertex/edge counts match; a 3-hop query returns |
| P2 agent loop | 16h | HHG-006 produces a valid, sane answer file end to end |
| P2.5 backtest | 6h | Verdict accuracy + Brier on a 200-case dev sample |
| P3 exam run | 6h | 20/20 files pass `validate.py`, 20/20 in graph |
| P4 UI | 4h | One page, case timeline visible |
| P5 submission | 6h | Repo, video, blog, social |

---

## Phase 0a — De-risk first (do this before anything else, 2h)

Account/infra problems are the classic hackathon time sink and they're discoverable in an hour.

- [ ] Savanna workspace created and reachable, **or** CE installed locally — whichever answers first
- [ ] Credentials in `.env`, `.env.example` committed, `.env` gitignored
- [ ] `tigergraph/tigergraph-mcp` installed, `discover_tools` returns
- [ ] One trivial GSQL round trip through MCP (create a vertex, read it back, delete it)
- [ ] Anthropic + Voyage API keys verified with a one-line call each
- [ ] **Measure the free-tier ceiling**: load 50k transactions, check storage used, extrapolate to
      590,742 + ~590k `NEXT` edges. If it won't fit, decide the subgraph cut here, not in Phase 3
      (ARCHITECTURE.md §9)

**Exit gate:** a GSQL query returns a row through MCP. Until then nothing else matters.

## Phase 0b — Data prep (4h, blocks everything downstream)

- [ ] Load `transactions.csv` with the projected core columns (`analysis/profile_dataset.py` has
      the list) → parquet. Do this once; never re-parse 675 MB of CSV in a loop
- [ ] Reconstruct `card_key` = `(customer_id, card1..card6)` for all transactions
- [ ] Anchor real `card_id` (`C01234-K1`) from `case_pack.csv` + `closed_cases_history.csv` onto
      the reconstructed keys, for both datasets, before graph load
- [ ] Build `NEXT` chain + `secs_since_prev` + `amt_pctile_for_card` at load time
      (ARCHITECTURE.md §7.3) — **as-of/expanding**, so no case sees its own future
- [ ] Precompute `DeviceProfile.is_full` and `global_card_count` (ARCHITECTURE.md §7.2)
- [ ] Run the deterministic R5 card-testing detector, store hits (ARCHITECTURE.md §7.4)
- [ ] Load `identity.csv`, join on `TransactionID`
- [ ] Load `closed_cases_history.csv` as `ClosedCase` + `INVOLVES`/`ON_CARD` edges
- [ ] Chunk + embed: fraud policy, the 5 typologies, **FinCEN SAR narrative guidance**
      (the narrative is graded against it) → `PolicyChunk` with vector attribute
- [ ] Embed `ClosedCase.analyst_notes` verbatim — do not LLM-paraphrase first

**Exit gate — if any of these differ, the load is wrong, stop and fix:**
13,553 customers · 14,893 cards · 590,742 txns · 144,432 identity · 5,565 closed cases ·
0 card_keys mapping to >1 card_id. `python analysis/profile_dataset.py` checks all of them.

## Phase 1 — Schema + MCP wiring (6h)

- [ ] Finalize GSQL schema from ARCHITECTURE.md §3
- [ ] Loading jobs for each vertex/edge type
- [ ] `src/graph/interface.py` — the ~12 allowed queries as typed functions
- [ ] `src/graph/local.py` — parquet implementation of the same interface (this is the backtest
      engine and the outage fallback, not a nice-to-have; ARCHITECTURE.md §1)
- [ ] `src/graph/tigergraph.py` — GSQL implementation, same signatures
- [ ] Contract test: both backends return identical results for all 12 queries on 5 sample cases
- [ ] Confirm `generate_gsql` works for one NL→GSQL round trip (useful in the demo; not in the
      scoring path)
- [ ] Wire `MultiServerMCPClient` into a bare LangGraph node as a smoke test

**Exit gate:** vertex/edge counts match Phase 0b's numbers, and a 3-hop traversal
(customer → card → txn → device) returns for HHG-006.

## Phase 2 — Core investigation loop (16h — highest score weight, spend the most time here)

Build the boring deterministic pieces first; they make the LLM's job smaller and they're what the
validator leans on.

- [ ] `src/policy/actions.py` — route table + `should_create_case` + `should_file_report`
      (ARCHITECTURE.md §7.1), with unit tests
- [ ] `src/policy/rules.py` — R1–R10 as predicates returning `(applies, actions, citation)`
- [ ] `src/io/answer.py` — state → `cases/<case_id>.json`
- [ ] `src/io/validate.py` — **the 12 invariants in PROJECT.md §4**. Write this early; it catches
      whole classes of silent zero-scores while there's still time to fix them
- [ ] Instrumentation decorator: `tool_calls`, `tokens`, `latency_s` on every tool/LLM call
      (required output fields — retrofitting these is painful)
- [ ] `Investigate` node (ARCHITECTURE.md §6)
- [ ] `Gather Evidence` node — rings, baselining, R5 check, typology deepening
- [ ] `Assess Uncertainty` node — rubric-anchored, ≥2 independent signals to leave the 0.15–0.85
      band, emits the signals it rested on
- [ ] `Gather More Evidence` + conditional edge back, loop cap 3
- [ ] `agent/simulate.py` — the single honest place evidence-request responses are simulated
- [ ] `Take Action` — `initial_actions` on first pass, `final_actions` on last; `interrupt()` for
      L1/L2
- [ ] `Explain` — summary (2–6 sentences) + FinCEN-structured SAR narrative when filing
- [ ] `Update Case Memory` — write back, embed, `SIMILAR_TO` edges
- [ ] **Read-back test**: run a second case and confirm the first is retrievable from the graph.
      Case memory that's written but never read is a demo claim, not a feature
- [ ] **End-to-end on HHG-006** (customer_report, risk 0.25, 97th-pctile amount, New device, no
      prior history) — the "weak score, real signal" archetype

**Exit gate:** HHG-006 produces a file that passes `validate.py` and reads sensibly to a human.

## Phase 2.5 — Backtest (6h — do NOT skip; it's the only accuracy signal before submission)

- [ ] `src/eval/backtest.py` — replay protocol with the as-of cutoff (ARCHITECTURE.md §8):
      hide the case row, its txns, and every case opened later
- [ ] Stratified dev sample (~200 from Jul–Sep, balanced across pattern × outcome), local backend
- [ ] Metrics: verdict accuracy · pattern confusion matrix · action-combo exact match ·
      SAR precision/recall · exposure error · **Brier score + reliability curve**
- [ ] Compare against the trivial baseline ("always fraud" scores 83.8% on this set) — if the
      agent isn't clearly beating that *and* calibrated, the loop needs work, not the UI
- [ ] Iterate prompts against the dev sample; log each run's numbers in the notes below
- [ ] Single holdout run (~150 from Oct+) at the end of the build. Once. Record it and stop

**Exit gate:** dev numbers recorded, and at least one prompt iteration demonstrably improved them.

## Phase 3 — Exam run (6h)

- [ ] Run all 20 `case_pack.csv` cases end to end
- [ ] `validate.py` passes 20/20
- [ ] Verify 20/20 actually written to the graph (query them back; `written_to_graph` must reflect
      a real write returning success)
- [ ] Reconcile against `analysis/exam_triage.csv`: any case where the agent's verdict contradicts
      the cheap features needs a human read of the trace before it ships
- [ ] Distribution sanity: verdict split shouldn't be 20-0, `FILE_REPORT` shouldn't appear on a
      third of the pack (history rate is 7.1%), `card_testing` shouldn't appear at all without
      unusually strong support (ARCHITECTURE.md §7.4)
- [ ] Check the cautions: device-profile trap avoided? `addr1`-null cases handled? Low `risk_score`
      customer reports not dismissed?
- [ ] Read all 20 traces in LangSmith. Prioritize HHG-006 / 014 (weak score, real signal),
      HHG-013 / 017 / 018 (likely false alarms), HHG-011 / 018 (five-figure histories),
      HHG-015 (three stacked signals)

**Exit gate:** 20/20 valid, 20/20 in graph, and you'd defend each verdict out loud.

## Phase 4 — UI (4h, minimal, cut first if time is short)

- [ ] Single page reading case state + evidence + decisions from TigerGraph
- [ ] Shows: case timeline, evidence log, probability, initial vs. final actions, approval route
- [ ] One visible `interrupt()` → human approve → resume, for the demo
- [ ] Hard stop at 4h. Do not let this eat Phase 2/2.5

## Phase 5 — Submission package (6h)

- [ ] Repo cleaned, README explains setup from clone to 20 files
- [ ] `cases/` complete, `validate.py` green, committed
- [ ] 3–5 min demo video. Script it around three cases: one clear fraud, one that needed more
      evidence and changed its recommendation, one legitimate that the agent correctly cleared.
      The middle one is the whole point of the competition — give it the most time
- [ ] Technical blog post — the 6 required points. Best material: the card_key reconstruction, the
      1,671→266 ring filter, the calibration curve, and what the backtest revealed
- [ ] Social post tagging @TigerGraphDB
- [ ] Optional monitoring track — only if Phase 3 is fully green (4,209 Nov–Dec txns score >0.7);
      separate folder, counts toward Innovation not accuracy

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Savanna free tier won't hold 590k txns + 590k edges | Medium | High | Measured in P0a; fall back to a documented subgraph (case cards' full history + Nov–Dec window) |
| MCP / cloud outage near the deadline | Medium | Fatal | `graph/local.py` produces the 20 files without TigerGraph (ARCHITECTURE.md §1) |
| Agent says "fraud" on everything | **High** | High | Half the pack is meant to be legitimate; history's 83.8% base rate is misleading. Caught by P2.5 calibration, not by reading outputs |
| Prior-case retrieval returns template-similar noise | High | Medium | Hybrid retrieval: structural filter first, vector rank second (ARCHITECTURE.md §3) |
| Ring detection fires on high-volume customers | **Confirmed** | Medium | Observed on HHG-018/007; require other cards to be anomalous too (ARCHITECTURE.md §7.2) |
| Missing required output field silently zeroes a part | Medium | High | `validate.py` written in P2, run before every commit of `cases/` |
| `initial` vs `final` actions collapse into one | Medium | Medium | Captured by graph topology at first/last Take Action pass, not reconstructed in Explain |
| UI/demo polish eats investigation time | High | High | P4 capped at 4h and cut first; 50% of score is P2/P2.5 work |
| Time sunk re-deriving dataset facts | Medium | Low | PROJECT.md §3 + `analysis/profile_dataset.py` |

---

## Notes / detours log

> Append dated entries whenever the plan changes, so a return-to-project pass can see why.

- **2026-09-20** — Dataset profiled end to end (`analysis/profile_dataset.py`, `README.md`).
  Resolved all four of ARCHITECTURE.md's open design questions from measurement: policy table,
  ring-filter thresholds (1,671→266), `NEXT`/baseline precomputation, R5 detector (28 cards
  book-wide, none at an exam case's flagged moment). Added the output contract + 12 validator
  invariants to PROJECT.md §4, the closed-case backtest (ARCHITECTURE.md §8) as Phase 2.5, and
  the cold-start triage table (`analysis/exam_triage.csv`). Deadline still not recorded.
