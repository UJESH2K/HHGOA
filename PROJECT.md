# PROJECT.md — TigerGraph Agentic Fraud Investigation (HHGOA)

> Read this file first, every time you (or Claude Code) return to this project after a detour.
> It is the source of truth for *what* we're building and *why*. ARCHITECTURE.md covers *how*.
> PLAN.md covers *what's next*.

---

## 1. What we're building

An AI agent that investigates fraud cases using TigerGraph, following this loop for every case:

```
Trigger → Investigate → Gather Evidence → Assess Uncertainty
   ↻ (loop back for more evidence if needed)
   → Take Action → Explain → Update Case Memory
```

The core thing being tested is **not** "detect fraud yes/no" — `risk_score` already exists for that.
It's: **given ambiguous, partial evidence, decide intelligently whether to act now, ask for more
evidence, or escalate to a human** — and do it inside real policy/permission constraints, with a
defensible, explainable case record.

---

## 2. Competition facts

- **Event:** TigerGraph Hacker House Goa (HHGOA), Agentic Fraud Investigation track
- **Required:** TigerGraph (Savanna or Community Edition) for graph + vector storage, GSQL +
  graph algorithms, TigerGraph MCP, GraphRAG, a UI to show investigation/case progression
- **Optional:** any agent framework, any LLM, external tools/data
- **Support:** TigerGraph Discord (`discord.gg/7JMkCAy9D3`), Devanshu (DevRel) +91 7404313376

### Judging weights (design priority order)
| Criterion | Weight |
|---|---|
| Investigation accuracy | 25% |
| Next best action (incl. uncertainty handling) | 25% |
| Agentic design and engineering | 15% |
| Innovation | 15% |
| Case summary & explainability | 10% |
| Demo quality & completeness | 10% |

**Takeaway:** 50% of the score is "did the agent reason well under uncertainty," not "did it use
cool tech." Don't let engineering polish crowd out investigation quality.

### Submission checklist
- [ ] Working agent, GitHub repo
- [ ] `cases/<case_id>.json` for all 20 benchmark cases — exact schema in §4
- [ ] All 20 cases also written to the graph (`written_to_graph: true` must be *true*, not aspirational)
- [ ] SAR narrative included wherever `FILE_REPORT` appears in final actions
- [ ] Next-best-action + approval route recorded **before** requesting more evidence **and** **after**
- [ ] 3–5 min demo video, end to end
- [ ] Technical blog post: what we built / architecture / how TigerGraph is used / agentic
      capabilities / what we learned / what we'd improve
- [ ] Social post (X or LinkedIn) tagging @TigerGraphDB, linking the blog/demo

---

## 3. Dataset facts (measured, not assumed)

Source: IEEE-CIS Fraud Detection (Vesta Corp), remixed for this task. **Never use the public
Kaggle files** — IDs/times/amounts are transformed here; recovering outcomes via the public
dataset is disqualifying.

Every number in this section is reproduced by `analysis/profile_dataset.py` (~3 min, pure pandas).
Full column-level detail lives in `README.md`.

| File | Rows | What it is |
|---|---|---|
| `transactions.csv` | 590,742 | Jul 2 – Dec 31 2016. 393 Vesta cols + `customer_id`, `ts`, `channel`, `risk_score`. **No fraud label.** |
| `identity.csv` | 144,432 | Device/connection records, online txns only, joins on `TransactionID` |
| `closed_cases_history.csv` | 5,565 | Finished investigations Jul–Nov 2016. **Only ground truth in the dataset.** |
| `case_pack.csv` | 20 | The exam — Nov–Dec 2016, entirely after the closed-case window (no leakage) |

### Entity model
```
Customer (13,553)
   └── Card (14,893 distinct, reconstructed — see below)
         └── Transaction (590,742, unique TransactionID)
               ├── identity record (online only, 144,432)
               ├── BillingRegion (addr1, 332 codes)
               └── EmailDomain (60 distinct across P_/R_)

ClosedCase (5,565) ──INVOLVES──> 14,955 txns ──ON_CARD──> 1,913 card_ids / 1,892 customers
```

### The `card_id` problem — must reconstruct, don't skip this
`transactions.csv` has no `card_id`. Case files key on `C01234-K1` format.
**Verified working key:** `(customer_id, card1, card2, card3, card4, card5, card6)` — 14,893
distinct cards, and **0 of 1,956** reachable keys map to more than one `card_id`. 5,529 of 5,565
closed cases keep all their transactions on a single key (the 36 that don't are plausibly genuine
multi-card episodes — worth a look). The K-suffix is NOT recoverable from card columns and is NOT
ordered by volume (of 61 two-card customers, the higher-volume card was K2 in 43 and K1 in 17) —
**anchor it** from `case_pack.csv` / `closed_cases_history.csv`, never guess it.

### Data cautions — in priority order (these will burn points if ignored)
1. `risk_score` is an input, not a label. Mean 0.171. 6 of 8 customer-report exam cases score
   < 0.40 (one is $482 at 0.25). An agent anchored to the score misses these.
2. **Shared device profiles are mostly noise** — 4,793 profiles appear on >1 card, largest is
   1,023 cards of all-null fields. Use the calibrated filter in ARCHITECTURE.md §7.2, not a
   naive "shared value" rule. Naive: 1,671 Nov–Dec candidates. Filtered: 266.
3. `id_15 = New` is 43% of all identity records — not evidence alone. **11 of the 20 exam cases
   carry it** (3 are `Found`, 6 are in-person with no identity record). It cannot be what
   separates them.
4. `card_id` must be reconstructed and anchored (see above).
5. Closed history is 83.8% confirmed fraud — that's a filtered population of alerts already
   worth investigating, **not the real-world base rate**. Don't calibrate `fraud_probability` on it.
   Note the July cleared-rate is 34% vs ~10% Aug–Oct; the history is not stationary either.
6. Nulls are meaningful, not zeros. `addr1` null on `C`-product txns is 94.8% (vs <2% elsewhere) —
   7 of 20 exam cases have no `addr1` on the flagged txn, ruling out the out-of-region pattern
   for that txn specifically.
7. Policy R10 (`BLOCK_ALL_CARDS`) needs 2+ cards with confirmed fraud — mean cards/customer is
   1.10, so check the customer actually has 2 cards before reaching for it.
8. Any evidence claim naming a `V1`–`V339` column must say it's an unnamed Vesta model feature —
   honesty here is explicitly scored.

### Closed-case outcome shape (this is what "good" looks like)
| Outcome | Count |
|---|---|
| `confirmed_fraud` | 4,665 (83.8%) |
| `cleared` | 900 (16.2%) |

| Action combo | Count |
|---|---|
| `CREATE_CASE\|BLOCK_CARD` | 4,268 |
| `VERIFY_WITH_CUSTOMER\|CLOSE_NO_FRAUD` | 900 |
| `CREATE_CASE\|BLOCK_CARD\|FILE_REPORT` | 397 |

Only three action combos ever appear, and `report_filed = Yes` on just 397 of 5,565 (7.1%).
Most confirmed fraud is a **single transaction** (median `n_txns` = 1). `exposure_usd` median
$117.09; cleared cases are always $0. Pattern mix: CNP 1,404 · ATO 1,205 · CNP-new-device 1,076 ·
out-of-region 955 · **card_testing 16** · **undocumented 9** · none 900.

**Read all 9 `undocumented` analyst notes by hand.** They are the highest-value text in the
dataset and retrieval will almost never surface them (9 in 5,565).

### ⚠ The closed cases cannot be used as supervised training data

Measured 2026-09-21, and it overturned a planned approach. **Every one of the 900 cleared cases
has `risk_score` ≥ 0.81.** Confirmed fraud spans 0.01–0.98. So below 0.81 the history contains
**0 legitimate examples and 4,038 fraud examples**.

This is not a data bug — it is what "closed case" means. A cleared case is an alert that fired
and was dismissed, and alerts fire on high scores; confirmed fraud also arrives via customer
reports, which carry low scores. Realistic sampling, fatal consequence:

- A classifier fitted on this set learns **`risk_score` < 0.81 ⟹ fraud**, which is the inverse
  of how the score actually works and pure sampling artifact. A first attempt scored AUC 0.991 /
  Brier 0.026 — entirely this leak.
- **17 of the 20 exam cases sit below 0.81**, in the region where the training data has never
  seen a legitimate outcome. The model would call all 17 fraud with near-total confidence, and
  be catastrophically wrong on the half the brief says are legitimate.
- Inside the only comparable stratum (`risk_score` ≥ 0.81, n=1,527, 41% fraud), the surviving
  features are themselves inverted in suspicious ways — `device_new` AUC 0.18, `profile_is_full`
  0.28 — so that stratum is not obviously trustworthy either, and it covers just 3 exam cases.

**Consequence:** `fraud_probability` comes from a transparent, hand-specified rubric with weights
grounded in measured base rates, **not** from a model fitted to closed-case labels. The closed
cases remain what the brief calls them — case memory for retrieval and pattern vocabulary — and
are used for that, plus qualitative backtesting. They are not a training set.

Reproduce: `python -m src.scoring.diagnose`.

### The closed cases are a labelled dev set — with the caveat above
The answer key for the 20 is hidden, but 5,565 labelled investigations are not. Replay closed
cases as if they were fresh alerts (feed the agent the first fraud txn + customer, hide the
outcome) and score verdict / pattern / action-combo against truth. Suggested split by open date:

| Split | Window | Cases | Confirmed | Cleared |
|---|---|---|---|---|
| Dev (tune on this) | Jul 1 – Sep 30 | 4,193 | 3,437 | 756 |
| Holdout (touch once) | Oct 1 – Nov 6 | 1,372 | 1,228 | 144 |

This is the only way to know whether a prompt change helped before submitting. Build it in Phase 2,
not Phase 4.

### Exam cases (`case_pack.csv`) — key facts
- 20 cases: 11 `risk_score` trigger, 8 `customer_report`, 1 `analyst_request`
- **16 of 20 customers already have closed-case history — retrieve it, it's free evidence**
  (C02354: 20 prior cases, C09933: 19, C13171: 19, C11923: 11)
- `risk_score` in the case pack is blank for non-risk-score triggers — always pull the real score
  from `transactions.csv`, never treat blank as "no signal"
- HHG-011 (10,361 customer txns) and HHG-018 (7,091) — any "unusual for this cardholder" logic
  must survive five-figure histories as well as 36-transaction customers
- Good first end-to-end test case: **HHG-006** — customer_report, risk_score 0.25, no prior case
  history, $482.12 at the 97th percentile of its own card's history, New device. Exactly the
  "weak score, real signal" archetype the competition is built around.

### Cold-start triage table (`analysis/exam_triage.csv`)

Cheap deterministic features for all 20, computed before any LLM call. **This is not an answer
key** — it is the baseline the agent's own findings should be reconciled against. If the agent
concludes "fraud" on a case whose amount sits at the 15th percentile of its own card's history
with no other signal, something is wrong upstream.

| Case | Amt | Ch | Risk | Hist | Amt %ile | New region | New product | id_15 | Proxy | Prior fraud |
|---|---|---|---|---|---|---|---|---|---|---|
| HHG-001 | $77 | in_person | 0.61 | 358 | 0.46 | — | — | — | — | 4 |
| HHG-002 | $292 | online | 0.79 | 35 | **1.00** | — | — | — | — | 1 |
| HHG-003 | $49 | in_person | 0.40 | 980 | 0.28 | — | — | — | — | 5 |
| HHG-004 | $128 | online | 0.34 | 209 | **0.99** | — | — | New | — | 3 |
| HHG-005 | $100 | online | 0.54 | 90 | 0.23 | — | — | New | — | 3 |
| HHG-006 | $482 | online | 0.25 | 101 | **0.97** | — | — | New | — | 0 |
| HHG-007 | $112 | in_person | 0.87 | 2431 | 0.71 | — | — | — | — | 18 |
| HHG-008 | $56 | online | 0.38 | 899 | 0.68 | — | — | Found | — | 18 |
| HHG-009 | $30 | online | 0.28 | 46 | 0.30 | — | — | Found | — | 0 |
| HHG-010 | $1000 | online | 0.90 | 33 | **1.00** | — | — | New | — | 0 |
| HHG-011 | $131 | online | 0.39 | 10306 | **0.97** | — | — | New | — | 10 |
| HHG-012 | $31 | in_person | 0.55 | 910 | 0.25 | — | — | — | — | 1 |
| HHG-013 | $36 | online | 0.76 | 1431 | 0.15 | — | — | New | — | 3 |
| HHG-014 | $75 | online | 0.05 | 71 | 0.79 | — | — | New | **ANONYMOUS** | 0 |
| HHG-015 | $600 | online | 0.77 | 68 | **1.00** | **yes** | — | New | — | 2 |
| HHG-016 | $60 | online | 0.37 | 48 | 0.75 | — | — | New | — | 0 |
| HHG-017 | $100 | online | 0.57 | 52 | 0.21 | — | — | Found | **HIDDEN** | 0 |
| HHG-018 | $39 | in_person | 0.48 | 5860 | 0.18 | — | — | — | — | 19 |
| HHG-019 | $100 | online | 0.90 | 219 | 0.40 | — | — | New | — | 3 |
| HHG-020 | $125 | online | 0.52 | 82 | 0.59 | — | **yes** | New | — | 2 |

What the table says, and what it doesn't:

- **Six cases sit at or near the top of their own card's amount history** (002, 004, 006, 010, 011,
  015). That is the single most discriminating cheap feature in the pack — more so than
  `risk_score`, which ranks 013 (0.76) above 006 (0.25).
- **HHG-015 is the only new-region case and also 100th percentile and New device** — the one case
  with three independent signals stacked.
- **HHG-014's tell is the anonymous proxy**, not the score (0.05, the lowest in the pack). It is
  the analyst-request case, and the brief hints it needs cross-card work.
- **HHG-013, 017, 018, 012, 005 sit low on their own amount distribution.** Low amount + high score
  is the classic false-alarm shape. Half the pack is meant to be legitimate; this is where to look.
- The table has no view of sequence, connected cards, or merchant fit. Those need the graph, and
  they are where the actual points are.

### Card testing is probably not in this exam pack
Under the literal R5 reading (3+ online authorizations under $5 on one card within 1 hour),
**28 cards in the entire 590k-row book** qualify, 9 of them in Nov–Dec, and **none at the flagged
moment of any exam case**. Loosening to <$10 gives 93 cards, <$25 gives 343. So: implement the R5
detector deterministically (it's cheap and exact), run the threshold sweep, but do not expect
`card_testing` to be the answer for many of the 20 — and be suspicious of an agent that returns it
often. Only 16 of 5,565 closed cases carry that pattern.

---

## 4. Output contract (scoring-critical — "missing fields score zero for that part")

One file per case: `cases/<case_id>.json`, 20 files. Full field docs in the brief
(`drive-download-.../README.md`, Answer Format section). The shape, condensed:

```jsonc
{
  "case_id": "HHG-006",
  "case": {
    "status": "open|closed_fraud|closed_legitimate|escalated",
    "verdict": "fraud|legitimate|uncertain",
    "fraud_probability": 0.0,           // scored for CALIBRATION, be honest
    "pattern": "card_testing|card_not_present_fraud|card_not_present_new_device|
                out_of_region_use|account_takeover|undocumented|none",
    "pattern_description": "",          // REQUIRED iff pattern == undocumented
    "affected_txn_ids": [], "first_suspicious_txn_id": "",
    "connected_card_ids": [], "connected_device_profiles": [],
    "exposure_usd": 0,
    "evidence": [{ "claim": "", "source": "graph|document|customer|external",
                   "ref": "", "entity_ids": [] }],
    "similar_prior_cases": ["CC-0141"],
    "summary": "", "written_to_graph": true, "graph_case_id": ""
  },
  "evidence_requests": [{ "type": "customer_validation|step_up_auth|analyst_info",
                          "asked_after_step": 1, "assumed_response": "" }],
  "next_best_actions": {
    "initial": [{ "action": "", "route": "auto|L1|L2", "reason": "R1: ..." }],
    "final":   [{ "action": "", "route": "auto|L1|L2", "reason": "R2: ..." }],
    "what_changed": "nothing"
  },
  "sar": { "file": false, "reason": "", "narrative": "", "subjects": [],
           "total_amount_usd": 0, "activity_dates": [] },
  "stop_reason": "",
  "tool_calls": 0, "tokens": 0, "latency_s": 0.0
}
```

**`tool_calls`, `tokens` and `latency_s` are required fields.** They have to be instrumented into
the agent from the first node, not reconstructed at the end. Design for this in Phase 2.

### Machine-checkable invariants (build `src/validate.py` in Phase 2, run before every submission)
1. All 20 files exist, parse, and carry every field above.
2. `sar.file == ("FILE_REPORT" in [a.action for a in final])` — the brief states they must agree.
3. `verdict == "legitimate"` ⇒ `affected_txn_ids == []`, `exposure_usd == 0`, `sar.file == false`.
4. `exposure_usd` **recomputed** from `affected_txn_ids` must match to the cent.
5. Every ID (`txn`, `card`, `customer`, `CC-xxxx`) exists in the dataset. Made-up IDs score zero.
6. `pattern == "undocumented"` ⇒ `pattern_description` non-empty.
7. Route of every action matches the policy table (§ ARCHITECTURE 7.1) given that case's exposure.
8. `evidence_requests == []` ⇒ `final == initial` and `what_changed == "nothing"`.
9. `sar.file == false` ⇒ narrative `""`, subjects `[]`, amount `0`, dates `[]`.
10. `sar.activity_dates` = [min, max] date of `affected_txn_ids`; `total_amount_usd` = `exposure_usd`.
11. `BLOCK_ALL_CARDS` present ⇒ customer demonstrably holds ≥2 cards (R10).
12. Calibration sanity across the set: if every `fraud_probability` is >0.8 or the verdict split is
    20-0, that is a bug, not a finding. Roughly half the pack is meant to be legitimate.

---

## 5. Key decisions log (append here whenever a real decision is made)

| Date/stage | Decision | Reasoning (short) |
|---|---|---|
| Initial | TigerGraph native vector (TigerVector) over external vector DB | One GSQL query does ANN + graph traversal together; stronger "how TigerGraph is used" story |
| Initial | Official `tigergraph/tigergraph-mcp` (not community forks) | 50+ tools incl. `generate_gsql`, `discover_tools`, `get_workflow`; has LangGraph adapter examples |
| Initial | LangGraph for orchestration | Flow is a stateful graph with a real loop (assess → gather more → reassess), not linear; native `interrupt()`/`Command(resume=)` matches approval requirements exactly |
| Initial | Claude Sonnet 5 (main reasoning) + Haiku 4.5 (cheap sub-tasks) | Tiered model usage — don't burn the expensive model on bulk classification/summarization across 590k txns / 5,565 cases |
| Initial | Voyage AI embeddings, stored as TigerGraph vertex attributes | Embed once at ingest (policy doc, 5 typologies, case summaries); no separate sync pipeline |
| Initial | Deterministic Python policy/approval engine, LLM only recommends | Brief explicitly separates recommend vs. execute; LLM can't also be the authority on what's authorized |
| Initial | TigerGraph GDS (Connected Components/Louvain, PageRank) for ring detection | Naive "shared device" rules flag half the book (see caution #2) — need graph-algorithmic filtering |
| Initial | Minimal UI (single page, built last) | Team priority is agent accuracy over UI polish |
| Initial | LangSmith tracing | Free explainability/audit trail of every node + tool call + interrupt |
| Data pass | `(customer_id, card1..card6)` as the card key, K-suffix anchored not guessed | Verified: 0 ambiguous mappings across 1,956 keys reachable from closed cases |
| Data pass | Ring filter = full profile + ≤10 global cards + ≥2 cards + ≤14d window | Cuts Nov–Dec candidates 1,671 → 266; naive rule would fire on half the book |
| Data pass | Backtest on closed cases (Jul–Sep dev / Oct+ holdout) before touching the 20 | Hidden answer key; this is the only pre-submission accuracy signal available |
| Data pass | Feature/graph layer must run without TigerGraph (parquet fallback) | Savanna or MCP outage cannot be allowed to block producing 20 answer files |
| 2026-09-21 | **Rejected** fitting a calibrated classifier on closed cases | All 900 cleared cases have risk_score ≥ 0.81; below that the history is 100% fraud. 17/20 exam cases live there. A fitted model scores AUC 0.991 by learning the sampling artifact and would call all 17 fraud. See §3 |
| 2026-09-21 | `fraud_probability` from a transparent weighted rubric instead | Weights grounded in measured distributions, not fitted to a biased label set. Auditable, and the reasoning is citable as evidence — which the fitted model's coefficients were not |

---

## 6. If you get lost

Re-read this file, then ARCHITECTURE.md for the how, then PLAN.md for what's next and what's done.
Don't re-derive dataset facts from scratch — they're in §3 above, in the full `README.md` dataset
guide, and reproducible via `python analysis/profile_dataset.py`.
