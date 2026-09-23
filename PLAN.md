# PLAN.md — Build Plan

> What's next, and what's done. Check items off as you go; add dated notes to the log at the
> bottom when a phase's plan changes. Re-read PROJECT.md and ARCHITECTURE.md before resuming
> after any gap.

**Deadline: 24 September 2026, 23:59 IST. One submission, by the team lead, no resubmissions.**
Submit at https://forms.gle/yxXzqSULGgZ9VUF56 Estimates below are in focused
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

- **2026-09-23 (later) — REAL DATASET IN, 20 REAL ANSWER FILES, FULL BOOK IN TIGERGRAPH.**

  The organizers' 708 MB `transactions.csv` arrived. `features/build.py` passed all five exit
  gates exactly on first run — 13,553 customers / 14,893 cards / 590,742 txns / 144,432 identity
  / 5,565 closed cases, 28 cards with an R5 burst, 9,706 device profiles. Every number the
  previous sessions documented from the friend's paraphrase is confirmed on the real data.

  `cases/` holds 20 answer files, `validate.py` reports **0 errors, 0 warnings**. Verdict split
  6 legitimate / 9 uncertain / 5 fraud; reports filed on 3 of 20 (15%).

  The whole book is loaded into Savanna with every vertex count exact: 590,742 Transaction,
  575,849 NEXT_TXN, 144,432 ON_DEVICE, 128,852 SHARES_DEVICE. Load took 649s.

  ### Reading the real README corrected three things

  The policy engine matched R1-R10, the routing table and both 3a gates exactly - the paraphrase
  was accurate. But:

  1. **The case pack column is `trigger_text`, not `narrative`.** Every trigger claim in the
     evidence log would have been silently empty.
  2. **The inferred `should_block_card` gate was wrong twice over.** It cited "Policy 4", which is
     the definition of *exposure*, not an action rule - a fabricated citation. And the policy
     deliberately does not authorise a block on the bank's own assessment: BLOCK_CARD comes from
     R2, R5 and R10 only, because R1 exists precisely to stop a bank blocking a legitimate
     customer on its own suspicion. What the policy *does* authorise without approval is asking
     (section 5). So a confident finding with nothing protective attached now recommends
     VERIFY_WITH_CUSTOMER and lets R2 or R3 settle it - which is also the shape of the brief's own
     worked example, and the reason `initial` and `final` differ.
  3. **R6 was filing regulatory reports on shared device *usage*.** The rule says "several cards
     show FRAUD from the same device profile". A moderate link establishes that other cardholders
     used the same model of phone, not that they were defrauded, so it now names the element,
     opens a case and monitors connected cards - but does not file.

  ### The ring filter was throwing away the one ring the pack asks about

  The biggest accuracy finding of the project. HHG-014's analyst trigger reads "several cards this
  month show purchases from the same unusual device profile", and that profile -
  `SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080` - is fully
  specified, sits on **52 cards belonging to 52 different customers** with 1-3 transactions each,
  and clusters into two tight bursts (15 Aug - 4 Sep, 14 Nov - 4 Dec).

  The `global_card_count <= 10` ceiling made it invisible, and the case came back `legitimate` at
  probability 0.04. The ceiling was calibrated to suppress generic fingerprints and it does that
  well - `Windows | Windows 10 | chrome 63.0 | 1920x1080` covers 842 customers - but it was also
  suppressing the real ring *because the real ring is big*.

  Measured: the obvious discriminators do not separate them. Transactions per customer is 2.19 for
  the ring and 2.18-3.05 for every generic profile. Total span is 111 days against 162-183. A
  specific device model plus build plus browser plus resolution on 52 unrelated customers is
  genuinely ambiguous, and a sharper threshold would be fitting a number to one case. So the link
  is now **graded** - strong below 10 cards, moderate to 60, not evidence above - and a moderate
  link needs three different customers in the window before it counts.

  ### Fixture lessons (all real bugs in the test scaffold)

  - Filler customers shared the builder's RNG stream, so adding filler changed the shape of all
    20 cases. Filler is now generated last, from its own stream.
  - Filler was generated as card-present, which produces no identity record - so it never
    populated the device profiles, leaving the "generic" profile on only the 20 case cards where
    the filter read it as a real ring.
  - The subscription amount was drawn from the same range as routine spending, so routine charges
    landed inside the detector's amount tolerance and made a regular subscription look irregular.
    The detector was right; the fixture was wrong to create it.
  - `expected_verdict` asserted `legitimate` for subscriptions. R7 prescribes a **restraint**
    ("do not block"), not a verdict, so the test now asserts the restraint.

- **2026-09-23 — TIGERGRAPH IS LIVE. The investigation runs on GSQL.**

  Savanna workspace reached, `FraudInvestigation` graph created alongside the workspace's
  pre-existing `Transaction_Fraud` sample graph (860,141 payment transactions, untouched), engine
  **4.2.5**, 16 GiB. `TG_HOST` was found by `python -m src.graph.savanna`, not by hunting the UI.

  **All 20 cases run end to end through `--backend tigergraph`**: 215 live GSQL queries, 20/20
  answer files, `validate.py` green, and **all 13 contract tests pass** - the TigerGraph backend
  and the parquet backend return identical answers.

  ### What first contact cost, so nobody pays it twice

  Every one of these was found on 1,563 synthetic rows rather than on the real 590,742:

  | # | Finding |
  |---|---|
  | 1 | Vertex/edge types are **database-global**, not graph-local. `Card` already existed, owned by the sample graph -> ours is `PaymentCard`. Checked all 9 of ours against their 18+22; that was the only collision |
  | 2 | **`Case` is a reserved GSQL keyword** -> `FraudCase` |
  | 3 | **`proxy` is a reserved identifier.** A bare `proxy STRING` attribute fails with "expecting compress/default/nullable/primary" reported at the NEXT comma - it took an isolation test to find |
  | 4 | `DEFINE FILENAME f = "ANY:x.csv"` resolves **server-side** into `/home/tigergraph/tigergraph/app`, which the engine refuses as a sensitive directory |
  | 5 | Column-by-name refs (`$"customer_id"`) are rejected unless the FILENAME is initialised with such a path. Positional is the only way through - so the loading job is now **generated** from the CSV headers by `src/graph/loadjob.py`, which is what stops positional from meaning fragile |
  | 6 | **`header="true"` is not honoured for REST-uploaded files** (the loader calls the source `Online_POST`). Every vertex type came out exactly +1 because the header row loaded as data. `deploy.py` strips it before upload |
  | 7 | `CREATE LOADING JOB` and `CREATE QUERY` are not idempotent -> drop-then-create, and `CREATE OR REPLACE QUERY` throughout |
  | 8 | **A `LIST` query parameter cannot be iterated or measured** (`TYP-1001: of type list parameter is invalid to call any function`). The cosine-in-an-accumulator retrieval saved as a DRAFT query that could not be installed, and surfaced as a **404 from REST** rather than an install error. Retrieval is now structural in GSQL, ranking moved to `features/casesim.py` shared by both backends, and true ANN uses TigerVector's built-in `vectorSearch` in `gsql/vector_upgrade.gsql` - which takes a query vector properly because it is a built-in, not user code iterating a parameter |

  ### Four divergences the contract test caught that nothing else would

  These are the reason the two-backend contract exists:

  - **`card_testing_episode` returned a different dict shape per backend.** The TigerGraph path had
    no `small_txn_ids` key, so `gather()` would have **raised KeyError at runtime** the first time
    a card-testing case ran through the graph.
  - **`known_regions` were `"204.0"` on parquet and `"204"` on the graph.** `addr1` is a float
    column in pandas; the float string was already reaching the answer files as *"Billing region
    204.0 has never appeared on this card"*. The graph was right.
  - **`recurring_charge` claimed "the card has no prior history"** about cards with hundreds of
    transactions. The two backends hand `detect()` different things - parquet passes the whole
    history, the graph passes only the amount-filtered survivors - so an empty input was ambiguous.
    It now takes `n_prior_txns` explicitly (`txn_seq` on the graph side, free).
  - **`similar_prior_cases` returned 3 on parquet and 0 on the graph.** The channel/pattern/exposure
    filters are *relaxable* - `casesim.narrow` keeps each only if k candidates survive - but the
    GSQL query applied them hard, leaving nothing to relax. The query now returns the as-of
    visible pool and the ranker narrows.

  Still outstanding: **the organizers' dataset**. Every credential is in place and verified
  (Anthropic key + Opus 5 confirmed live, Voyage, TG host/secret/user, Savanna control-plane key).

- **2026-09-21 (new machine, third session)** — **TigerGraph layer written, not yet executed.**
  The submitter has no working endpoint and is likely to stand up their own instance later, so
  this was written blind against pyTigerGraph and GSQL v2 and is explicitly flagged as unverified
  in `src/graph/tigergraph.py`. ~1,800 lines:

  - `gsql/schema.gsql` — 9 vertex types, 15 edge types. Three things precomputed as attributes
    rather than queries, each for a measured reason: `amt_pctile_prior` (one property read instead
    of a 10,332-row scan for C11923), the `NEXT` chain (burst evidence in one hop), and
    `DeviceProfile.is_full` + `global_card_count` (the ring filter as a property lookup).
  - `gsql/queries_core.gsql` — the entity and behaviour queries, one per `GraphBackend` method.
  - `gsql/queries_ring.gsql` — **the file where the graph earns its place.** An iterative
    connected-component walk from one seed card over the *filtered* `SHARES_DEVICE` edge set,
    plus in-component degree so a hub can be told from a leaf. The filter lives in the edge set:
    without it, components over shared-device links return one component holding half the book.
  - `gsql/queries_memory.gsql` — hybrid retrieval (structural filter, then cosine in an
    accumulator), GraphRAG over the policy corpus, case read-back, and cross-case entity
    recurrence.
  - `gsql/loading.gsql` + `src/graph/export.py` — the loading jobs and the CSV export that builds
    the three things that exist nowhere until export runs: the NEXT chain, the filtered
    card-to-card `SHARES_DEVICE` edges (same-customer pairs excluded — a household with two cards
    on one laptop is not a ring), and the exploded `INVOLVES` links. `--subgraph` writes the
    documented relevance cut for a free workspace: case-involved cards at full history plus all of
    Nov–Dec, no silent sampling.
  - `src/graph/tigergraph.py` — the backend. Two queries deliberately return candidate rows and
    let `features/recurring.py` and `features/testing.py` do the sequence arithmetic, so both
    backends share one implementation of each rule and the contract test can actually prove
    agreement.
  - `src/graph/deploy.py` — staged install (`--check` before anything expensive) plus `--measure`
    for the free-tier ceiling decision.
  - `tests/test_tigergraph_contract.py` — 13 tests asserting the two backends return identical
    answers. Skipped without `TG_HOST`; this is what makes writing a backend blind recoverable,
    since the first live run will find mistakes and they will surface as disagreements rather
    than as quietly wrong answer files.

  Embeddings stored as `LIST<DOUBLE>` with cosine in a GSQL accumulator rather than TigerVector's
  native vector type: the native version is faster and the better story, but the syntax varies by
  engine version and no engine is available to check against. Swapping is two queries and two
  attribute declarations. Decision recorded rather than hidden.

  `--backend tigergraph` is wired into the runner with **no silent fallback** — a run that quietly
  used parquet while reporting TigerGraph would make the write-up false.

- **2026-09-21 (new machine, second session)** — **Phase 2 milestone reached: the chain runs end
  to end and produces 20 valid answer files.** Not on the real dataset, which is still not on this
  machine — on `src/fixtures/generate.py`, a synthetic book with the organizers' schema and the
  same case ids, deliberately fed through the *real* `features/build.py` so it exercises the card
  reconstruction, the as-of baselines, the ring filter and the R5 detector rather than a parallel
  path. `python -m src.agent.run --data data_fixture` investigates all twenty in 0.4s: 215 graph
  queries, 0 model calls, 85% of evidence items from a graph query, and
  `python -m src.io.validate cases_fixture` prints "All invariants pass."

  New: `src/agent/run.py` (the seven stages), `src/agent/simulate.py` (the one place a response is
  invented and disclosed), `src/fixtures/generate.py`, `tests/test_end_to_end.py`. 231 tests.

  Four more defects, all found by running the chain rather than reading it:
  **(a)** `ring_signals` merged on a key that was both index and column, and raised on every case
  with a device record; **(b)** the validator treated `_run_manifest.json` as an answer file and
  reported 33 structure errors about a file that was never an answer; **(c)** a `legitimate`
  verdict could coexist with BLOCK_CARD, because R2 fires on any dispute — the rubric now holds
  the verdict at `uncertain` unless something *explains* the dispute (R7 recurrence, or a
  withdrawal), and `evaluate()` withdraws the block when the verdict is legitimate;
  **(d)** a device profile shared across three different customers was classified
  `account_takeover`, which is precisely backwards — takeover is one account falling under
  someone else's control, and cross-account sharing now outranks it as `undocumented`.

  Decisions confirmed by the submitter: Opus 5 for the graded run (`.env` updated), LangGraph for
  orchestration with direct Anthropic SDK model calls, a properly designed UI, and yes to both the
  autonomous-monitoring track and the undocumented-typology discovery work.

  **Top blocker is now the TigerGraph endpoint.** `TG_HOST` is empty and every hostname variant of
  the workspace id returns NXDOMAIN. GSQL, graph algorithms, MCP and GraphRAG — four judged
  requirements, ~30% of the score between them — can be written blind but not executed until the
  Savanna Connect panel gives up a resolvable host.

- **2026-09-21 (new machine)** — `STRATEGY.md` added: gap analysis against the judging criteria,
  the seven differentiators we are betting on, a measured cost plan, and the decisions still
  needing a call. Phase 1 of that plan is done and green (173 tests, 31 still dataset-gated):
  the rubric scorer, deterministic pattern classification, the value-of-information evidence
  selector, the shared policy-context builder, and the cost/token meter. `analysis/rubric_on_pack.py`
  runs the whole deterministic layer over all 20 exam cases from the committed triage CSV, with no
  dataset, keys or graph.

  Five defects fixed, two of them consequential and both found by running the pack rather than by
  reading code: **R2 could never fire on a customer-report trigger** (8 of 20 cases recommended no
  protective action), and **no rule covered a confident fraud finding that arrived by risk score**
  (HHG-015: $599.94 at probability 0.90, recommendation "create a case"). The new
  `should_block_card` gate closes the second, with brakes for R3, R5 and R7 — but its threshold is
  inferred from the closed-case action distribution, not quoted from the Fraud Policy, because the
  policy document ships with the dataset and the dataset is not on this machine. **Reconcile it
  against the policy text before the graded run.**

  Consequence now on the critical path: with R2 firing correctly, every customer-report case
  blocks the card, and R7 is the only brake. The recurring-charge detector (same merchant,
  comparable amount, regular interval, from the card's own history) is therefore no longer
  optional — it is what separates blocking on every dispute from recognising a cardholder
  disputing their own subscription.

- **2026-09-20** — Dataset profiled end to end (`analysis/profile_dataset.py`, `README.md`).
  Resolved all four of ARCHITECTURE.md's open design questions from measurement: policy table,
  ring-filter thresholds (1,671→266), `NEXT`/baseline precomputation, R5 detector (28 cards
  book-wide, none at an exam case's flagged moment). Added the output contract + 12 validator
  invariants to PROJECT.md §4, the closed-case backtest (ARCHITECTURE.md §8) as Phase 2.5, and
  the cold-start triage table (`analysis/exam_triage.csv`). Deadline still not recorded.
