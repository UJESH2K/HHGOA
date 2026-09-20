# ARCHITECTURE.md — TigerGraph Agentic Fraud Investigation

> The "how." Read PROJECT.md first for the "what and why."

---

## 1. Stack

| Layer | Choice | Notes |
|---|---|---|
| Graph + vector store | **TigerGraph** (Savanna or CE), native vector type (TigerVector) | Vectors live as a vertex attribute; ANN search + graph traversal in one GSQL query. No separate vector DB. |
| MCP server | **`tigergraph/tigergraph-mcp`** (official, v1.0.0+) | 50+ tools: CRUD, raw GSQL, loading jobs, `generate_gsql` (NL→GSQL), `discover_tools`, `get_workflow`. Has a LangGraph `MultiServerMCPClient` adapter out of the box. |
| Orchestration | **LangGraph** | State machine with loops; `interrupt()` / `Command(resume=...)` for human approval; checkpointer (`PostgresSaver` for anything beyond local dev) persists case threads. |
| Primary LLM | **Claude Sonnet 5** | Main reasoning/tool-calling loop, explanations. |
| Secondary LLM | **Claude Haiku 4.5** | Cheap bulk tasks: summarizing prior cases, classifying/labeling at scale. |
| Embeddings | **Voyage AI** (`voyage-3` family) | Embed fraud policy chunks, 5 typology docs, and short case-summary strings once at ingest; store as vertex attributes. |
| Policy / approval | **Deterministic Python rules engine** | Not the LLM. Parses the fraud policy into an action → approval-route table. LLM proposes, this gates/executes. |
| Graph algorithms | **TigerGraph GDS**: Connected Components / Louvain, PageRank / degree centrality | Ring detection on filtered `SHARES_DEVICE` / `SHARES_CARD` edges — never a naive "any shared value" rule. |
| Observability | **LangSmith** (free tier) | Automatic trace of every node, tool call, interrupt — doubles as the explainability record. |
| UI | Minimal single-page dashboard, built last | Reads case state straight from TigerGraph. Not a priority; agent quality is. |

### Hard constraint: the graph is a dependency, not a single point of failure

The deliverable is 20 JSON files. A Savanna outage, an expired workspace, or an MCP regression the
night before submission cannot be allowed to produce zero of them. So:

- **Every graph query used in the investigation loop has a local parquet/pandas equivalent**, behind
  one interface (`src/graph/`, see §5). TigerGraph is the default backend and the one demoed; the
  local backend is the fallback and what the dev-set backtest runs against (it is also 100× faster
  for 5,565 replays).
- `written_to_graph` in the output is set by the *actual* write call returning success, never
  hardcoded `true`.
- This is an engineering-quality point too, not just insurance: the same interface is what makes
  the backtest in §8 cheap enough to run on every prompt change.

---

## 2. High-level flow

```
Trigger (risk_score | customer_report | analyst_request)
   │
   ▼
[Investigate]  — create/open Case, resolve entities, pull free evidence (prior cases),
   │              bounded recent-txn window, device context if online, typology GraphRAG
   ▼
[Gather Evidence] — deepen: ring detection (GDS), amount/behavior baselining, policy lookup
   │
   ▼
[Assess Uncertainty] — confidence score, risk level, "enough to act?" decision
   │
   ├─(not enough)──► [Gather More Evidence] ──► loop back to Assess Uncertainty
   │                  (controlled, policy-approved: ask customer, step-up auth, ask analyst)
   │
   ▼ (enough)
[Take Action] — recommend via policy engine; execute only if authorized, else interrupt()
   │             for human approval
   ▼
[Explain] — evidence used / why more evidence was requested / why these actions
   │
   ▼
[Update Case Memory] — write case + outcome back to graph for future retrieval
```

Each box below is a LangGraph node. State is a single typed object threaded through all of them
(see §4) — nodes only ever *append* to `evidence_log` and `decision_log`, never overwrite.

**`next_best_actions.initial` is captured at the first pass through Take Action**, before any
evidence request is issued. `final` is captured at the last pass. If the loop never fires, they are
equal and `what_changed` is `"nothing"`. Getting this wrong silently zeroes part of the score, so
it is a property of the graph topology, not something the Explain node reconstructs afterwards.

---

## 3. TigerGraph schema

### Vertices
| Vertex | Key facts | Notes |
|---|---|---|
| `Customer` | 13,553 | `customer_id` |
| `Card` | 14,893 (reconstructed) | key = `(customer_id, card1..card6)`, `card_id` field anchored from case/history data, not guessed |
| `Transaction` | 590,742 | Core ~19 cols projected out of 397; `risk_score`, `ts`, `channel`, `amt_pctile_for_card` (precomputed, see §7.3) |
| `DeviceProfile` | ≤9,706 | `DeviceInfo\|OS\|browser\|screen`; carries `is_full` and `global_card_count` so the ring filter is a property lookup, not a subquery |
| `BillingRegion` | 332 | `addr1` |
| `EmailDomain` | 60 | purchaser + recipient domains, union |
| `ClosedCase` | 5,565 | outcome, pattern, exposure_usd, action_combo — **this is training/memory data** |
| `Case` | grows with use | the live case object this agent creates and progresses |
| `PolicyChunk` | small, fixed | embedded fraud-policy / typology / regulator text chunks, vector attribute for GraphRAG |

### Edges
```
Customer -[OWNS]-> Card
Card -[USED_IN]-> Transaction
Transaction -[NEXT]-> Transaction        # ordered by ts, within a card key — build at load time,
                                          # this is what makes card-testing/burst patterns 1-hop
Transaction -[HAS_IDENTITY]-> IdentityRecord   # online only
Transaction -[ON_DEVICE]-> DeviceProfile       # online only
Transaction -[BILLED_TO]-> BillingRegion
Transaction -[FROM_DOMAIN / TO_DOMAIN]-> EmailDomain
Card -[SHARES_DEVICE {profile, first_ts, last_ts}]-> Card   # derived, filtered — see §7.2
Customer -[HAS_CASE]-> ClosedCase
ClosedCase -[INVOLVES]-> Transaction
Case -[ABOUT]-> Customer, -[ON_CARD]-> Card, -[TRIGGERED_BY]-> Transaction
Case -[SIMILAR_TO {score}]-> ClosedCase   # written by GraphRAG/similarity retrieval
```

### Vector-bearing vertices
- `PolicyChunk.embedding` — fraud policy + 5 typology docs + selected FinCEN/FFIEC narrative
  guidance, chunked. The SAR narrative is graded against FinCEN's standard; put that guidance in
  the retrievable corpus, not just in a prompt.
- `ClosedCase.summary_embedding` — the `analyst_notes` field embedded directly (they are already
  well-written prose; do not paraphrase them through an LLM first and lose the detail).
- Optionally `Case.summary_embedding` on the live case, updated as evidence accrues.

### Retrieval quality note
`analyst_notes` are highly templated — most differ only in IDs, amounts and dates. Pure vector
similarity over them will return near-random neighbours with uniformly high scores. **Retrieve
hybrid:** filter structurally first (same pattern candidates, similar exposure band, same channel,
overlapping device/region), then rank by vector similarity within that set. Retrieving 5 prior
cases that merely share a sentence template is worse than retrieving none — it manufactures false
confidence and pollutes `similar_prior_cases`, which is a scored field.

---

## 4. LangGraph state object (contract between nodes)

```json
{
  "case_id": "HHG-006",
  "trigger": { "type": "customer_report", "flagged_txn_id": "3476682", "narrative": "..." },
  "customer": { "customer_id": "C07297", "n_cards": 2 },
  "card_key": "C07297::...", "card_id": "C07297-K1",
  "flagged_txn": { "...": "..." },
  "risk_score": 0.25,
  "prior_cases": [ { "case_id": "CC-...", "outcome": "...", "pattern": "..." } ],
  "recent_txns": [ "...bounded NEXT-chain window..." ],
  "device_context": null,
  "candidate_typologies": [ { "typology": "account_takeover", "similarity": 0.81 } ],
  "ring_signals": [],
  "baseline": { "amt_pctile": 0.97, "new_region": false, "new_product": false },
  "fraud_probability": null,
  "sufficient_evidence": null,
  "loop_count": 0,
  "pending_evidence_requests": [],
  "evidence_requests": [],
  "initial_actions": [],
  "final_actions": [],
  "approval_status": {},
  "evidence_log": [ { "stage": "investigate", "claim": "...", "source": "graph",
                      "ref": "query:...", "entity_ids": [] } ],
  "decision_log": [],
  "status": "open",
  "meta": { "tool_calls": 0, "tokens": 0, "started_at": 0.0 }
}
```

`evidence_log` entries use the **same shape as the output contract's `evidence` objects**
(`claim` / `source` / `ref` / `entity_ids`). Serialization is then a copy, not a transformation —
one less place to drop a required field.

`meta` is incremented by a decorator on every tool call and every LLM call. `tool_calls`, `tokens`
and `latency_s` are required output fields; instrument them at the wrapper, on day one.

---

## 5. Module layout

```
src/
  graph/
    interface.py      # the ~12 queries the agent is allowed to make, as typed functions
    tigergraph.py     # GSQL implementation (default)
    local.py          # parquet/pandas implementation (fallback + backtest speed)
  features/
    cards.py          # card_key reconstruction + card_id anchoring (PROJECT.md §3)
    baseline.py       # per-card amount percentile, new region/product, velocity
    rings.py          # device-profile filter + connected components (§7.2)
    testing.py        # deterministic R5 card-testing detector (§7.4)
  agent/
    nodes.py          # the 7 LangGraph nodes
    prompts/          # one file per node, version-controlled, diffable
    state.py          # the typed state object above
  policy/
    actions.py        # action enum, route table, SAR trigger — pure functions, no LLM (§7.1)
    rules.py          # R1–R10 as predicates over state
  io/
    answer.py         # state -> cases/<case_id>.json serializer
    validate.py       # the 12 invariants in PROJECT.md §4
  eval/
    backtest.py       # replay closed cases, score verdict/pattern/actions (§8)
cases/                # the 20 deliverable JSON files
analysis/             # profile_dataset.py + derived CSVs
```

The point of `graph/interface.py` being a closed set of ~12 functions: the agent is not given raw
GSQL-generation as a tool in the scoring path. Free-form NL→GSQL is a great demo moment and a bad
way to get reproducible answers on 20 graded cases. Use `generate_gsql` interactively while
developing queries, then freeze the good ones into the interface.

---

## 6. Node design notes

### Investigate
1. Resolve identity (`customer_id` / `card_key` / `card_id`) from whatever the trigger gives you.
2. Create the `Case` vertex, link to `Customer` / `Card` / flagged `Transaction`.
3. Pull prior `ClosedCase` history first — cheapest, highest-value evidence (free for 16/20 exam customers).
4. Pull a bounded recent-transaction window via the `NEXT` chain, not full history.
5. Pull device/identity context only if `channel = online`.
6. Always resolve `risk_score` from `Transaction`, never trust a blank in the trigger payload.
7. Compute the cheap baseline (`amt_pctile`, `new_region`, `new_product`) — deterministic, no LLM.
8. One hybrid GraphRAG query (structural filter + vector rank) for candidate typologies.
9. Emit the state object, `evidence_log` seeded with stage-1 entries.

### Gather Evidence
- Ring detection: §7.2 filter, then GDS Connected Components over the surviving `SHARES_DEVICE` edges.
- Behavior baselining: compare the flagged txn against the card's own distribution. **Percentile,
  never raw count** — it has to mean the same thing for a 36-txn card and a 10,306-txn card.
- Card-testing check: §7.4, deterministic, runs regardless of what the LLM suspects.
- Typology deepening: targeted GraphRAG follow-up only if probability is still mid-band.

### Assess Uncertainty
Produces `fraud_probability` (0–1) and `sufficient_evidence` (bool). This node is worth more
iteration than any other — it drives both the accuracy score and the calibration score.

**Scoring method: LLM-judged against an explicit rubric, anchored by deterministic priors.** Not a
free-form number, and not a hand-tuned linear model either. Concretely: the prompt carries the
measured base rates (PROJECT.md §3), the evidence log, and a rubric that names what each signal is
worth *and what it is not worth* — `id_15=New` is 43% of the book and moves probability very
little on its own; a top-percentile amount on the card's own history moves it a lot; a rare shared
device inside 14 days moves it a lot; `risk_score` alone moves it barely at all. Have the node emit
the probability **and** the two or three signals it rested on, then assert:

- ≥2 independent signals required before probability may exit the 0.15–0.85 band (this is the
  brief's own stopping rule, §6 of the policy — enforce it in code rather than hoping).
- Probability inside 0.30–0.70 with exposure > $500 ⇒ cannot stop; must request evidence or escalate.
- Log the rubric inputs. "Explainability" is 10% of the score and this is where it's earned.

### Gather More Evidence (conditional loop target)
- Only fires policy-approved requests: `customer_validation`, `step_up_auth`, `analyst_info`.
- **Responses are not provided by the organizers.** Simulate them in a single, honest place
  (`agent/simulate.py`) and record the assumption verbatim in `evidence_requests[].assumed_response`.
  Do not let the LLM invent a convenient reply inline — that is both a scoring risk and, in the
  demo, indistinguishable from fabrication.
- Assume denial/confirmation *consistently with the evidence*: a customer-report trigger already
  means the customer disputes it (R2 applies without asking again).
- **Loop cap: 3.** On cap-out without sufficiency, go to Take Action with `verdict: uncertain` and
  record why in `stop_reason`. R8 then forces `ESCALATE_TO_ANALYST` if exposure > $500.

### Take Action
- Agent proposes ranked actions with reasons citing rule numbers.
- `policy/actions.py` assigns the route deterministically and splits `auto` (executable, stubbed)
  from `L1`/`L2` (recommended + `interrupt()`).
- First pass fills `initial_actions`; last pass fills `final_actions`.

### Explain
- Assembles `evidence_log` + `decision_log` into `case.summary` (2–6 sentences — the brief says
  keep it short; the evidence list carries detail) and, when `sar.file`, the narrative.
- **SAR narrative is generated against the FinCEN structure**: who / what / when / where / how /
  why suspicious, 6–12 sentences, standing alone without the case file. Retrieve the FinCEN
  narrative-guidance chunks for this generation rather than relying on the model's memory of them.

### Update Case Memory
- Write final state to `Case`, compute `summary_embedding`, link `SIMILAR_TO` for prior cases
  **actually used** in reasoning (not everything retrieved).
- Verify by reading back: run case 2's Investigate and confirm case 1 is retrievable. A memory
  that is written but never read is a demo claim, not a feature — and the brief explicitly asks
  that the next investigation be able to find it.

---

## 7. Resolved design decisions

These were the open questions. All four are now settled from measurement
(`analysis/profile_dataset.py`, sections C and D).

### 7.1 Policy engine — the complete table

Pure functions, no LLM, no exceptions. Actions and routes come verbatim from the Fraud Policy.

```python
ROUTE = {  # everything not listed is "auto"
  "DECLINE_TRANSACTION": "L1",
  "BLOCK_ALL_CARDS":     "L2",
  "FILE_REPORT":         "L2",
}
def route(action, exposure_usd):
    if action == "BLOCK_CARD":
        return "L1" if exposure_usd <= 2500 else "L2"
    return ROUTE.get(action, "auto")
```

`auto` (agent may execute): `ALLOW_TRANSACTION`, `MONITOR_CARD`, `MONITOR_CONNECTED_CARDS`,
`WARN_CUSTOMER`, `VERIFY_WITH_CUSTOMER`, `STEP_UP_AUTH`, `GENERATE_REPORT`, `CREATE_CASE`,
`ESCALATE_TO_ANALYST`, `CLOSE_NO_FRAUD`.

Two gates that are decided in code, never by the model:

```python
def should_create_case(state):          # policy 3a
    return (state.fraud_probability >= 0.30
            or state.evidence_requests
            or state.trigger.type == "customer_report")

def should_file_report(state):          # policy 3a — ALL of: confirmed/strong AND any trigger
    strong = state.verdict == "fraud" or state.fraud_probability >= 0.70
    return strong and (state.exposure_usd > 1000
                       or state.ring_signals            # shared device / region / other customer
                       or state.pattern == "undocumented")
```

R1–R10 become predicates in `policy/rules.py`, each returning `(applies: bool, actions: list,
citation: str)`. The citation string is what lands in the output's `reason` field, so rule numbers
are never hand-typed by the model.

Note what the history says about this: only three action combos ever appear across 5,565 closed
cases, and only 7.1% carry a report. An agent emitting `FILE_REPORT` on a third of the exam pack is
mis-calibrated regardless of how good its narrative reads.

### 7.2 Device-profile ring filter — calibrated

Measured distribution: 9,706 profiles, 4,856 fully specified (no nulls in any of the four fields).
Among full profiles the median is on 2 cards, p90 on 16, p95 on 35. The generic top of the
distribution is not a ring — `Windows | Windows 10 | chrome 63.0 | 1920x1080` covers 842 customers.

**The filter:**

```
is_full            # all four of DeviceInfo, id_30, id_31, id_33 present
AND global_card_count <= 10          # ~90th percentile of all profiles
AND cards_in_window >= 2
AND window_span_days <= 14
```

Effect on the exam window: **1,671 naive candidates → 266.** Both numbers are in
`profile_dataset.py` section C, so the filter's value is demonstrable in the write-up.

Two caveats that matter for the exam pack:

- **High-volume customers generate false rings.** Run on the 20, the filter surfaces HHG-018
  (C02354, 7,091 txns) nine times and HHG-007 (2,792 txns) four times, simply because a customer
  with thousands of transactions touches many devices. Require that the *other* cards on the shared
  profile also look anomalous, or normalize by the customer's device count, before calling it a ring.
- A ring signal is a claim about **other people's cards**. Under R6 it pulls in `FILE_REPORT` and
  `MONITOR_CONNECTED_CARDS`, which is expensive to get wrong. Set the bar high and name the shared
  element explicitly in the evidence, as R6 requires.

### 7.3 `NEXT` chain and precomputed baselines — build at load time

The chain is a load-time job, not a query-time sort:

```python
t = t.sort_values(["card_key", "ts"])
t["next_txn_id"] = t.groupby("card_key").TransactionID.shift(-1)
t["secs_since_prev"] = t.groupby("card_key").ts.diff().dt.total_seconds()
t["amt_pctile_for_card"] = t.groupby("card_key").TransactionAmt.rank(pct=True)
```

`amt_pctile_for_card` as a stored `Transaction` attribute is the single highest-value precomputed
feature in the build — it is what makes "unusual for this cardholder" one property read instead of
a full-history scan, and it is the feature that separates the exam pack best (PROJECT.md §3).
Compute it **expanding/as-of** (only over transactions before the one in question) for any feature
the agent reasons with, so a case is never scored using its own future.

### 7.4 Card-testing detector (R5) — deterministic, and rarer than expected

R5 is precise enough to implement exactly: 3+ online authorizations under $5 on one card within
one hour, followed by a larger purchase. Implemented as a sliding window over the `NEXT` chain.

Measured across the whole book:

| Small-amount threshold | Cards matching | With follow-up | Follow-up > $100 | In Nov–Dec |
|---|---|---|---|---|
| < $5 (literal R5) | 28 | 19 | 6 | 9 |
| < $10 | 93 | 53 | 29 | 18 |
| < $25 | 343 | 149 | 83 | 61 |

**No exam case has a qualifying sequence at its flagged moment**, at any of the three thresholds.
(C11923 — HHG-011's customer — has one in August, four months before its December alert; that is
prior history worth citing, not the current episode.) Closed-case history agrees: 16 of 5,565.

So: run the detector on every case because it is exact and nearly free, cite it when it fires,
and treat a `card_testing` verdict on the exam pack as a claim needing unusually strong support.

---

## 8. Evaluation harness (`src/eval/backtest.py`)

The hidden answer key means the only pre-submission accuracy signal comes from replaying labelled
closed cases as if they were fresh alerts.

**Replay protocol.** For a closed case, construct a synthetic trigger from its `first_fraud_txn_id`
(or, for `cleared` cases, its flagged transaction), then **hide everything at or after the case's
`opened_at`**: the case row itself, its `txn_ids`, and — critically — any other closed case opened
later. Otherwise the agent retrieves the answer it is being asked to produce. The as-of rule in
§7.3 exists for the same reason.

**Metrics**, aligned to what the judges weight:

| Metric | Against | Maps to |
|---|---|---|
| Verdict accuracy (fraud / legitimate) | `outcome` | Investigation accuracy 25% |
| Pattern accuracy, and confusion matrix | `pattern` | Investigation accuracy |
| Action-combo exact match | `actions_taken` (only 3 values exist) | Next best action 25% |
| SAR decision precision/recall | `report_filed` (7.1% positive) | Next best action |
| Exposure error (relative) | `exposure_usd` | Investigation accuracy |
| **Calibration: Brier score + reliability curve** | `outcome` | Explicitly scored field |

Calibration deserves its own line. `fraud_probability` is called out in the brief as scored for
calibration, and a reliability curve over a few hundred replays is the only way to know whether
"0.7" from this agent means anything. It is also a strong slide in the blog post.

**Budget discipline.** 5,565 replays × a full agent loop is not affordable. Run the dev split
(Jul–Sep, 4,193 cases) as a **stratified sample of ~200**, balanced across pattern and outcome,
on the local backend with Haiku for the non-reasoning nodes. Reserve the Oct+ holdout (1,372) for
a single end-of-build run on a ~150 sample. Touching the holdout repeatedly turns it into a dev set.

Watch the base rates while reading results: the history is 83.8% confirmed fraud and July alone is
34% cleared against ~10% later. An agent that says "fraud" every time scores 84% on this set and
would still fail an exam pack designed to be roughly half legitimate.

---

## 9. Remaining open questions

- [ ] Savanna free-tier limits: workspace size against 590,742 transaction vertices + ~590k `NEXT`
      edges + ~145k identity vertices. Measure early (PLAN Phase 0) — if it doesn't fit, load a
      filtered subgraph (all cases' cards + their full history + the Nov–Dec window) and say so.
- [ ] Whether `Transaction` carries all ~19 core columns or the V/C/D/M blocks stay in parquet and
      are fetched only on demand. Leaning: keep them out of the graph; nothing in the loop reads
      them except as honest "unnamed model feature" corroboration.
- [ ] Human-approval UX in the demo: real `interrupt()` with a human clicking approve, versus a
      scripted resume. Real is a better demo; scripted is safer on the clock. Decide by Phase 4.
- [ ] Whether to attempt the optional autonomous-monitoring track (4,209 Nov–Dec transactions score
      > 0.7). Innovation is 15%; this is the cheapest credible way to earn it *if* the core 20 are
      already solid. Gate it behind Phase 3 being complete.
