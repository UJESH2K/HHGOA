# STRATEGY.md — how we turn this repo into a winning submission

> Written for you (the person submitting), not for the judges. It is a briefing: what the
> competition actually grades, what is already built, what is missing, what I propose to build,
> what it will cost, and the decisions I need from you.
>
> Read sections 1, 4, 9 and 10 if you read nothing else. Section 10 is where I need your answers.
>
> Written 2026-09-21 on the new machine, after reading every file in the repo and running what
> could be run without the dataset.

---

## 1. TL;DR

**What exists:** a genuinely strong *foundation*. ~2,800 lines, 66 tests, and — more valuable —
a set of measured findings about the dataset that most of the other 300-400 teams will not have
(the card_id reconstruction, the 1,671 → 266 device-ring filter, the proof that the closed cases
cannot be used as training data). The deterministic half of the system — features, policy engine,
answer contract, validator — is done and tested.

**What does not exist:** *most of what the judges named as required*. There is no TigerGraph code,
no GSQL, no graph algorithms, no MCP, no GraphRAG, no embeddings, no agent loop, no UI, and no
answer files. Right now the repo would score close to zero, because the deliverable is 20 answer
files and there are none.

**The honest position:** we are about 35% through the work and most of the way through the hard
*thinking*. What remains is build work, and it is well specified by the docs already here.

**Where I think we win.** Not by having a better fraud model — the brief deliberately makes that
impossible. By being the team that:

1. Uses TigerGraph as the **reasoning engine** (GSQL + graph algorithms + native vectors in the
   scoring path), not as a place data happens to sit. Many teams will export to pandas and call it
   GraphRAG. That is the sponsor criterion and the easiest place to be visibly better.
2. Picks its next evidence request by **expected value of information** — it computes which
   question would most change the decision, instead of always asking the customer. This is
   directly the 25% "next best action under uncertainty" criterion.
3. Publishes a **measured cost and latency ledger** per case. "20 defensible investigations for
   under $2 and under 4 minutes total" is a claim with a number behind it, and cheap to prove.
4. Is **honest about calibration**, including the finding that the obvious ML approach is a trap.
   Judges remember the team that showed them a negative result.

---

## 2. What the brief actually grades

| Criterion | Weight | Where it is won | Status today |
|---|---|---|---|
| Investigation accuracy | 25% | evidence gathering + pattern ID + exposure | features done, no agent |
| Next best action (uncertainty) | 25% | policy engine + when to ask + updating the recommendation | policy engine done, no uncertainty layer |
| Agentic design & engineering | 15% | orchestration, tools, memory, permissions | nothing built |
| Innovation | 15% | graph + GraphRAG + agentic originality | nothing built |
| Case summary & explainability | 10% | the case record and its prose | contract done, no prose |
| Demo quality | 10% | video + UI | nothing built |

Two things follow, and they are the whole strategy:

- **50% of the score is uncertainty handling and evidence quality.** Not tech. An agent that
  correctly says "I don't know yet, and here is the one question that would settle it" beats an
  agent that confidently says "fraud" twenty times.
- **The required-components list (TigerGraph, GSQL, algorithms, MCP, GraphRAG, UI) is not scored
  directly — it is scored through Innovation and Agentic design (30% combined), and through being
  taken seriously at all.** Skipping any of them reads as not having finished.

---

## 3. What is already built (and is good)

Verified on this machine before I touched anything: **35 passed, 31 skipped** - the skips only
because the dataset is not here yet.

| Module | What it gives us | Why it matters |
|---|---|---|
| `src/features/cards.py` | reconstructs `card_id` from `(customer_id, card1..card6)`, anchors the K-suffix from the case files, refuses to guess | `transactions.csv` has no card column. Teams that guess will cite card IDs that do not exist and lose the ID-validity points |
| `src/features/baseline.py` | as-of amount percentile, new region/product, velocity | `amt_pctile_prior` is the best cheap signal in the pack — better than `risk_score` |
| `src/features/rings.py` | the calibrated shared-device filter | a naive "shared device" rule fires on half the book. This is a real differentiator |
| `src/features/testing.py` | exact R5 card-testing detector | cheap, exact, and it tells us card testing is nearly absent — so we will not over-claim it |
| `src/policy/actions.py` + `rules.py` | R1–R10 as pure predicates, routing table, the two policy-3a gates | the LLM recommends, this decides. Exactly the separation the brief asks for. 35 tests |
| `src/io/answer.py` | the only place answer JSON is built | "missing fields score zero for that part" |
| `src/io/validate.py` | 12 machine-checked invariants including cross-field agreements | catches silent zero-scores a human proofread never would |
| `src/graph/interface.py` | a closed set of ~14 typed queries, two backends, `as_of` enforced everywhere | lets TigerGraph be the default without TigerGraph being a single point of failure |
| `src/scoring/diagnose.py` | reproducible proof that fitting a model on closed cases is a trap | blog-post material and an accuracy save |

The docs (`PROJECT.md`, `ARCHITECTURE.md`, `PLAN.md`, `README.md`, `TRANSFER.md`) are unusually
good. I am not rewriting them; I am adding to them.

### Five real defects I found, all now fixed

The last two are the serious ones. They were only visible by running the whole deterministic
layer over the 20 exam cases, which is why `analysis/rubric_on_pack.py` now exists — one command,
no dataset needed, and it shows the probability, the pattern, the recommendation and the chosen
next question for every case in the pack.

1. **Card-testing transaction IDs were saved as a string, not a list.**
   `src/features/build.py` writes `small_txn_ids=ct.small_txn_ids.astype(str)`, which stores
   `"[3514030, 3514031]"`. `LocalBackend.card_testing_episode` handed that string straight
   back, so evidence built from it would carry a stringified list where transaction IDs belong —
   and `affected_txn_ids` is a scored field checked against the dataset. Parquet stores list
   columns natively, so the cast is unnecessary. *(Confirmed by reproducing it here.)*
2. **`similar_closed_cases(channel=...)` ignored `channel` entirely**, and did no similarity
   ranking at all, despite its docstring promising hybrid retrieval. `similar_prior_cases` is a
   scored field, and it would have been filled by exposure proximity alone. It now filters
   structurally (pattern, channel, exposure band, relaxing only when a filter would leave nothing
   to rank) and returns a `similarity` column, so the retrieval is citable. Closed cases carry no
   channel of their own, so `features/build.py` now derives one from each case's anchor
   transaction.
3. **The card-testing window was inconsistent between two implementations** —
   `features/testing.py` looked ±7 days, `graph/local.py` looked 7 days backwards only.
   Backwards-only is correct (a December alert must not cite an August burst as its current
   episode), so both now share one constant.

4. **R2 could never fire on a customer-report case.** The rule tested
   `customer_response == "denied"`, so a dispute that arrived *as the trigger* — which is how all
   eight customer-report cases in the exam pack arrive — left the rule silent. The cardholder had
   said the transaction was not theirs, and the agent recommended opening a case and nothing else.
   `ARCHITECTURE.md` §6 states the intended behaviour outright ("a customer-report trigger already
   means the customer disputes it, R2 applies without asking again") and `PolicyContext` already
   derived it; only the rule had not been told. **That is 8 of 20 cases recommending no protective
   action.**

5. **A confident fraud finding on a risk-score trigger recommended nothing protective.** R1–R10
   are every one of them conditional on a customer response or a specific pattern, so a case that
   reaches a confident verdict from graph evidence alone matches none of them. HHG-015 is the
   example: $599.94, probability 0.90, a billing region the card had never used, verdict fraud —
   and a recommendation to create a case and stop. The closed history says what should happen
   instead: 4,268 of 5,565 investigations end in `CREATE_CASE|BLOCK_CARD`, and the only ones that
   end without a block are the 900 cleared. There is now a `should_block_card` gate alongside the
   two policy-3a gates, with three brakes taken from rules that are in the policy (a
   confirmation under R3, a recurring charge under R7, and card testing under R5, which
   prescribes its own narrower action set and must not be overridden).

   **This one carries a caveat and it is in the code as well as here:** unlike R1–R10, that
   threshold is inferred from the action distribution in `closed_cases_history.csv`, not quoted
   from the Fraud Policy — because the policy document ships with the dataset and the dataset is
   not on this machine yet. It has to be reconciled against the policy text before the graded
   run, and if the policy words it differently, the policy wins.

Defects 1–3 were latent, in code paths nothing calls yet. Defects 4 and 5 would have cost real
points on 12 of the 20 cases, and neither was findable by reading the code — only by running it
over the pack and looking at what came out.

### And one consequence worth planning around

With R2 now firing correctly, **every customer-report case blocks the card**, because R7 is the
only brake and R7 needs a recurring-charge detector that does not exist yet. On the cheap
features alone that is right for HHG-004, 006 and 011 and questionable for HHG-003, 009 and 018,
whose amounts sit at the 28th, 30th and 18th percentile of their own card's history. So the
recurring-charge detector (same merchant, comparable amount, regular interval, from the card's
own history) has moved onto the critical path: it is the difference between an agent that blocks
on every dispute and one that recognises a cardholder disputing their own subscription.

---

## 4. What is missing (the gap list)

Ordered by how much score is sitting on it.

| # | Missing | Weight riding on it | Effort |
|---|---|---|---|
| G1 | **The agent loop** — 7 nodes, the evidence loop, instrumentation | 25% + 15% | 14h |
| G2 | **The uncertainty layer** — `fraud_probability` rubric + stopping rule + evidence selection | 25% | 6h |
| G3 | **TigerGraph** — schema, loading, GSQL queries, graph algorithms, writes | 15% + required | 10h |
| G4 | **GraphRAG** — chunk + embed policy / typologies / FinCEN + case notes, ANN retrieval in GSQL | 15% + required | 5h |
| G5 | **MCP wiring** — the official `tigergraph/tigergraph-mcp` as a real tool surface | required | 3h |
| G6 | **The 20 answer files** | the entire deliverable | 3h once G1–G2 land |
| G7 | **UI** — analyst console showing the case progressing | 10% | 5h |
| G8 | **Eval harness** — closed-case replay, calibration curve | protects 25% | 5h |
| G9 | **Submission package** — video, blog, social | 10% + required | 6h |

Total ≈ 57 focused hours. That is the real number; `PLAN.md` says the same in different words. If
we have less time than that, section 6 says what to cut and in what order.

---

## 5. How we stand out — seven specific things

Each is chosen because it is (a) cheap relative to its score impact and (b) unlikely to be in most
of the other submissions.

### D1 — TigerGraph as the reasoning engine, not a warehouse *(Innovation, Agentic design, sponsor)*

The failure mode most teams will hit: load the CSVs into TigerGraph, do the actual analysis in
pandas, and describe that as GraphRAG. We do the opposite, and make it visible:

- **All ~14 investigation queries are installed GSQL queries**, called by name. The parquet
  backend stays as the offline fallback and the backtest engine, and the contract test proving both
  return identical answers is itself a strong engineering exhibit.
- **Ring detection runs TigerGraph's graph algorithms** — Weakly Connected Components over the
  *filtered* `SHARES_DEVICE` edges to find the component, then degree / PageRank centrality to say
  which card is the hub, and Louvain when a component is big enough to have internal structure.
  The filter (full profile, ≤10 global cards, ≥2 cards, ≤14 days) is what makes this meaningful
  rather than noise — and the 1,671 → 266 number is the evidence.
- **Similarity runs inside GSQL** using TigerVector's native vector attributes: one query that
  structurally filters candidate prior cases and ANN-ranks them by embedding in the same traversal.
  That is the thing TigerGraph can do that a vector DB plus a graph DB cannot, and the reason it is
  worth saying out loud.
- **The live `Case` is a vertex with edges** (`ABOUT`, `ON_CARD`, `TRIGGERED_BY`, `SIMILAR_TO`,
  `EVIDENCED_BY`), written as the investigation progresses — not one JSON blob at the end. Case
  progression is then literally a subgraph, which is the best possible demo frame.

### D2 — Evidence requests chosen by expected value of information *(Next best action, 25%)*

The idea I most want to build. The brief asks the agent to "determine what additional evidence is
needed" and "decide when there is enough information to act". Nearly everyone will implement:
*if probability is mid-band, ask the customer.*

Instead: for each policy-approved request the agent could make (`customer_validation`,
`step_up_auth`, `analyst_info`), enumerate the possible responses, re-run the rubric under each,
and compute **how much the recommended action set would change**. Then ask the question with the
highest expected change — and record in the case file that the others were considered and why they
were not worth asking.

Why it is strong:

- It turns "ask for more evidence" from a reflex into a decision with a stated justification.
- It produces exactly the artifact the brief demands — next best action recorded *before* the
  request and *after* it — with a real reason for the delta rather than a restatement.
- It is deterministic and nearly free: it reuses the rubric, costs no extra LLM calls, and is
  fully explainable.
- It gives the demo its best 30 seconds: *"the agent decided asking the customer would not change
  the decision, and requested step-up auth instead — here is the arithmetic."*

### D3 — A case record that actually progresses *(Explainability 10%, Agentic design)*

Append-only `evidence_log` and `decision_log`, each entry stamped with the stage, the query that
produced it, and the entity IDs — already the shape the answer contract wants, so serialization is
a copy rather than a transformation. `what_changed` is generated by diffing the initial and final
action sets, not written by the model. Every recommendation carries its rule citation from the
policy engine, so rule numbers are never hallucinated.

### D4 — A measured cost and latency ledger *(Agentic design, and the thing you asked for)*

We instrument tokens, cost, tool calls and wall time per node from day one (they are required
output fields anyway), and publish a table: cost per case, cache hit rate, where the tokens went,
how much of the evidence was gathered deterministically versus by the LLM. Target headline:
**all 20 cases for under $2 and under 4 minutes, with 3 LLM calls per case.** Section 7 has the
arithmetic. The lever that makes it true is architectural, not a trick: the deterministic layer
gathers, the LLM only judges and writes.

### D5 — Discovering an undocumented typology *(Innovation, Investigation accuracy)*

The dataset README says outright: *"Not every fraud pattern present in the data is documented."*
The answer schema has an `undocumented` pattern with a required `pattern_description`, and only 9
of 5,565 closed cases use it. That is an invitation.

Plan: run community detection over the Nov–Dec window on the filtered shared-origin graph, extract
the structural signature of any community that does not match the five documented typologies, write
it up as a typology document, embed it into the `PolicyChunk` corpus, and let the agent retrieve
and cite it like any other typology. If it fires on a case, we have a defensible `undocumented`
verdict with a graph-derived description. If it fires on none, we say so and the work still earns
Innovation — it is a graph algorithm doing discovery rather than classification.

### D6 — Case memory we can prove is read, not just written *(Agentic design, brief requirement)*

Process the 20 cases **in chronological order of `opened_at`**, so later cases can retrieve earlier
ones. Log every time a retrieved case changed the probability or the action set, and report the
count. A memory that is written but never read is a claim; a memory with a logged effect is a
feature. Also: cross-case entity recurrence (the same device profile or billing region appearing in
two of our own cases) is a graph query, and it is exactly what the brief means by "identify
recurring fraud patterns, entities, and relationships across cases".

### D7 — Honest calibration, including the negative result *(Investigation accuracy, credibility)*

`fraud_probability` is scored for calibration. We already know that fitting a model on the closed
cases produces AUC 0.991 by learning a sampling artifact, and that 17 of the 20 exam cases live in
the region where the training data contains no legitimate outcomes at all. So: a transparent
rubric, weights grounded in measured base rates, the enforced rule that the probability cannot
leave the 0.15–0.85 band on a single signal, and a reliability curve from the closed-case replay on
the only stratum where both outcomes occur. Then we tell the judges about the trap. Very few
submissions will contain a negative result, and it is the most credible thing in the blog post.

---

## 6. Build order

Each phase ends in something submittable-or-better. The ordering is chosen so that a hard stop at
any phase boundary still leaves a valid submission.

| Phase | What | Hours | Ends with |
|---|---|---|---|
| **P1** | Fix the three defects · rubric scorer + VOI engine · instrumentation wrapper | 8 | `fraud_probability` for all 20, tested, no credentials needed |
| **P2** | Deterministic end-to-end: rubric → policy → answer → `cases/*.json`, templated prose | 6 | **20 valid files with zero external dependencies. This is the safety net** |
| **P3** | TigerGraph: schema, load, ~14 GSQL queries, WCC / Louvain / centrality, case writes, contract test | 10 | `--backend tigergraph` produces identical answers; cases readable back out of the graph |
| **P4** | GraphRAG: chunk + embed policy, 5 typologies, FinCEN narrative guidance, 5,565 analyst notes; hybrid ANN retrieval in GSQL | 5 | typology and prior-case retrieval grounded in the graph |
| **P5** | Agent loop: 7 nodes, real LLM reasoning, evidence loop, approval interrupts, MCP tool surface | 14 | the real agent; prose and judgement replace templates |
| **P6** | Eval: closed-case replay, calibration curve, one holdout run | 5 | numbers that tell us whether P5 helped |
| **P7** | UI: analyst console — timeline, evidence, uncertainty, initial vs final actions, approval button, cost meter | 5 | the demo |
| **P8** | Submission: 20 final files, video, blog, social | 6 | done |

**The one non-negotiable is the end of P2.** After that, everything is improvement rather than
rescue. If time collapses, cut in this order: P7 (UI → one static page), P6 (eval → 50 cases
instead of 200), P4 (embeddings → structural retrieval only), D5 (typology discovery).

**Do not cut P3.** TigerGraph is the sponsor and a stated requirement; a submission that runs on
parquet and mentions TigerGraph in the README will be marked as not having done the task.

---

## 7. Cost and performance plan

Real prices (first-party Anthropic API, verified against the current model table, not from memory):

| Model | Input $/1M | Output $/1M | Cache read | Use here |
|---|---|---|---|---|
| Claude Opus 5 | $5.00 | $25.00 | ~$0.50 | the 20 graded runs — quality where the score is |
| Claude Sonnet 5 | $2.00 | $10.00 | ~$0.20 | development iteration |
| Claude Haiku 4.5 | $1.00 | $5.00 | ~$0.10 | bulk backtest, summarising 5,565 notes |

The Batch API is 50% off for anything not latency-sensitive — the backtest sweeps qualify.

**Per-case budget** (3 LLM calls: assess, classify, explain + SAR; ~12k input of which ~80% is the
cached policy corpus and rubric; ~1.5k output each):

| Run | Model | Cost/case | 20 cases |
|---|---|---|---|
| Graded exam run | Opus 5 | ~$0.16 | **~$3.10** |
| Graded exam run | Sonnet 5 | ~$0.065 | ~$1.30 |
| Dev iteration | Haiku 4.5 | ~$0.02 | ~$0.40 |
| 200-case backtest | Haiku 4.5 + batch | ~$0.01 | ~$2.00 per sweep |

So the entire project — dozens of dev iterations, several backtest sweeps, and a handful of full
graded runs — lands comfortably under **$20**. Embeddings add cents.

**What makes this cheap, and what we will show:**

1. **Deterministic-first.** The features, the ring filter, the R5 detector, the baselines, the
   policy engine and the routing are all pure code. The LLM never scans 590k transactions; it reads
   a bounded, pre-digested evidence pack. Measurable claim: ~85% of evidence items in a case file
   come from a graph query, not from a model.
2. **Prompt caching on a frozen prefix.** The policy text, typologies, FinCEN guidance and rubric
   are byte-stable, so they cache; only the per-case evidence varies. Cache reads are about 10% of
   input price. We will report the measured `cache_read_input_tokens` ratio rather than claim it.
3. **Tiered models.** Opus 5 for the 20 that are graded, Haiku for the hundreds that are practice.
4. **Direct Anthropic SDK, not a LangChain LLM wrapper.** A recommendation and a change from the
   current plan: keep LangGraph for orchestration (its `interrupt()` maps exactly onto the approval
   requirement, and checkpointing gives resumable cases), but call the model through the Anthropic
   SDK directly. The wrapper hides `cache_control` placement and the usage fields the ledger needs
   — and the `langchain` installed here is 0.0.339, nearly two years stale.
5. **Latency.** The deterministic path is milliseconds and the graph queries are indexed lookups.
   The target is under 12 seconds per case end to end, which makes the demo watchable in real time
   instead of a montage of spinners.

---

## 8. TigerGraph plan, concretely

Because this is the sponsor criterion, here is exactly what goes in the graph and what it does.
(The schema is already specified in `ARCHITECTURE.md` §3; this is the execution and the emphasis.)

**Deployment: Savanna, not local Docker.** This machine has 15.2 GB RAM (1.1 GB free at the time of
checking), 28.7 GB free disk, and no Docker. TigerGraph CE wants 8 GB minimum plus ~15 GB of disk
for the image, which would be tight and would fight with everything else running. `TRANSFER.md`
says your `.env.local` already holds `TG_SECRET` and `TG_WORKSPACE_ID`, so a Savanna workspace
exists — we use it, with auto-start / auto-stop on as the brief requires.

**Load shape.** 590,742 transactions + ~590k `NEXT` edges + 144k identity records is the full book.
The first action in P3 is to measure the free-tier ceiling with a 50k-row load and extrapolate. If
it does not fit, the documented cut is: every card involved in the 20 exam cases and the 5,565
closed cases at *full* history, plus the entire Nov–Dec window. That keeps every graded traversal
exact and is honest in the write-up. No silent sampling.

**Graph algorithms in the scoring path** (not decoration):

- Weakly Connected Components over filtered `SHARES_DEVICE` → the ring's membership.
- Degree / PageRank centrality within a component → which card is the hub and which is a leaf.
  This is what distinguishes "a ring" from "a high-volume customer who touches many devices", the
  documented false-positive mode on HHG-018 and HHG-007.
- Louvain when a component is large enough to have internal structure → feeds D5.
- Bounded k-hop traversal on the `NEXT` chain → burst and sequence evidence in one hop.

**TigerVector** carries `PolicyChunk.embedding` (policy, the 5 typologies, FinCEN narrative
guidance, plus any typology we discover) and `ClosedCase.summary_embedding` (the analyst notes,
embedded verbatim — they are already good prose, and paraphrasing them through a model would lose
the detail that makes retrieval work). Retrieval is hybrid *inside one GSQL query*: structural
filter first (same channel, comparable exposure band, overlapping region or device), ANN rank
second. Pure vector similarity over these notes returns near-random neighbours with uniformly high
scores because the notes are templated — filtering first is what makes `similar_prior_cases` worth
citing.

**MCP.** The official `tigergraph/tigergraph-mcp` server, wired two ways: as the agent's graph tool
surface in an `--explore` mode (and in the demo, where watching the agent choose a graph tool is
the point), and as the authoring path during development via `generate_gsql`. The graded path calls
frozen installed queries by name, because 20 graded cases should not depend on what the model
improvises as a query — and I will say exactly that in the blog post rather than hide it. Both are
real uses of MCP; the split is a deliberate engineering decision, not a shortcut.

---

## 9. What I need from you (blockers)

| # | What | Why it blocks | How |
|---|---|---|---|
| B1 | **The dataset folder** `drive-download-20260919T105649Z-1-001/` (~704 MB) | It is not on this machine. Nothing can be verified against real data without it — and it contains the **organizers' README**, which is the document we are graded against | Copy it into the repo root (it is gitignored) or re-download from the organizers |
| B2 | **`.env`** with `TG_SECRET`, `TG_WORKSPACE_ID` (Savanna), `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` | Any live run | Copy `.env.local` from the old machine, or paste the keys. `.env` is gitignored — I checked |
| B3 | **A TigerGraph endpoint that resolves.** This is now the top blocker. `.env` has `TG_SECRET` and `TG_WORKSPACE_ID`, but `TG_HOST` is empty and every hostname variant of the workspace id returns NXDOMAIN — the note in `.env` says the same was true on 2026-09-21. Without a reachable endpoint, four judged requirements (GSQL, graph algorithms, MCP, GraphRAG) cannot be executed, only written | **P3, and it gates 30% of the score** | Open savanna.tgcloud.io → the workspace → **Connect** panel, start the workspace if stopped, and paste the host it shows into `TG_HOST` |
| B4 | **The deadline** | `PLAN.md` literally says "not recorded". It decides how much of section 6 we attempt | Tell me the date and time |
| B5 | **Rotate the keys if the old transcript was shared** | `TRANSFER.md` says they were pasted in plain text | Your call |

On B1 — if the dataset genuinely cannot be moved, tell me and I will build a synthetic fixture
generator (a few thousand rows with the identical schema) so the whole pipeline, the tests and the
TigerGraph load can be exercised without it. That is useful anyway: it makes the repo runnable by a
judge who does not have the 675 MB file. But it is not a substitute for the real answer files.

Meanwhile, nothing in P1 needs any of this, so I am starting there.

---

## 10. Decisions I need from you

My recommendation is first in each list. Tell me the ones you disagree with; the rest I will
proceed on.

1. **Model for the graded run.** *Recommend: Opus 5.* Half the score rides on reasoning under
   uncertainty, and the difference between Opus 5 and Sonnet 5 across all 20 cases is about $1.80.
   The repo's `.env.example` currently says Sonnet 5. Cheap insurance, your call.
2. **Orchestration.** *Recommend: LangGraph for the loop, Anthropic SDK for the model calls.*
   LangGraph's `interrupt()` is the cleanest implementation of "some actions require human
   approval", which is a stated requirement. The alternative — a ~200-line custom loop — is less to
   install and easier to instrument, but loses the approval primitive and the framework optics. See
   section 7, point 4.
3. **Scope of the UI.** *Recommend: one FastAPI page, read-only except a single approve / deny
   button* (FastAPI and uvicorn are already installed here). Capped at 5 hours. The alternative is
   a React dashboard, which is prettier and is where hackathon projects go to die.
4. **Do we attempt the optional autonomous-monitoring track?** (4,209 Nov–Dec transactions score
   above 0.7.) *Recommend: only if P6 is green.* It is the cheapest credible Innovation add-on, and
   also the classic way to lose a weekend.
5. **Do we build D5 (undocumented-typology discovery)?** *Recommend: yes, timeboxed to 4 hours.*
   It is the most distinctive thing on the list and the dataset README all but asks for it. But it
   is genuinely optional, and the second thing I would cut.
6. **How much do you want to be in the loop?** I can (a) work through P1 → P2 and show you 20
   answer files before touching anything else, (b) check in at every phase boundary, or (c) check in
   only when blocked. *Recommend (a)*, because seeing real output changes how you will want the
   rest built.

---

## 11. What is already done, before you answer anything

None of this needed the dataset, the keys, or your decisions. **Tests: 231 passed, 36 skipped**
(the 32 skipped still need the dataset). Everything below runs on a fresh clone.

- [x] All five defects in section 3, each with a regression test that names the bug it guards.
- [x] `src/scoring/rubric.py` — the transparent `fraud_probability` scorer. Weights grounded in
      measured base rates; the ≥2-independent-signals rule enforced in code rather than hoped for
      in a prompt; `id_15 = New` and `risk_score` given weights that make it *arithmetically
      impossible* for either to be the signal that licenses a verdict; a `profile_fit` family so a
      case can be cleared on consistency rather than only ever escalated; and an R3 override so a
      cardholder's confirmation cannot be out-voted by anomaly signals — except when the account
      itself looks taken over, in which case whoever answered may not be the cardholder.
- [x] `src/scoring/patterns.py` — deterministic pattern classification, every candidate keeping
      its rationale and its rejection reason. A card-present transaction cannot be labelled
      card-not-present (six exam cases), and `undocumented` cannot be reached without coordinated
      activity across accounts (R9 attaches a regulatory filing to it, and 9 of 5,565 closed cases
      used it).
- [x] `src/features/recurring.py` — the recurring-charge detector, R7's evidence and the only
      brake on R2. The dataset has **no merchant column**, so "same merchant" is not observable;
      the detector looks for the signature a subscription actually leaves — a near-identical
      amount, at a standard billing cadence, with the flagged charge landing where the next one
      was due — and its evidence claim says "recurring charge", never "merchant", because it
      cannot see one. Exposed on the graph interface as `recurring_charge`, so both backends
      must implement it.
- [x] `src/agent/voi.py` — the value-of-information selector (D2), working end to end. On
      HHG-006 it declines to ask anything and says why; on HHG-013 it prefers the cheap
      authentication challenge to a phone call and shows that passing would close the case; on
      HHG-018 it refuses all three options, two on policy grounds.
- [x] `src/agent/context.py` — the one place an assessment becomes a `PolicyContext`, shared by
      the real decision and the counterfactuals, so the selector cannot optimise for a decision
      the agent would not then take.
- [x] `src/agent/meter.py` — tokens, cost, cache hit rate, tool calls, latency, per stage and per
      model, plus the run-level roll-up that becomes the cost table in the blog post.
- [x] `analysis/rubric_on_pack.py` — the whole deterministic layer over all 20 exam cases from a
      committed CSV. **Run this first.** It is how defects 4 and 5 were found.
- [x] `analysis/profile_dataset.py` extended to emit the consistency columns the rubric needs
      (`known_region`, `known_email_domain`, `timing_typical`, prior-window counts).

- [x] **`src/fixtures/generate.py` + `src/agent/run.py` - the chain runs end to end.**
      A synthetic dataset with the organizers' schema and the same case ids, handed to the *real*
      build pipeline, then investigated by the real runner: 20 answer files, **0 errors, all 12
      invariants passing**, in 0.4 seconds. 215 graph queries, 85% of all evidence items from a
      graph query rather than a model. `tests/test_end_to_end.py` asserts the verdicts against
      planted ground truth - the subscription disputes clear under R7, the card-testing burst is
      detected, the cross-account ring files a report and pulls in the connected cards, and no
      legitimate verdict ever seizes a card.
- [x] `src/agent/simulate.py` - the single place an evidence-request response is invented, and
      the place it is disclosed. Every `assumed_response` carries the word SIMULATED, the
      probability it was chosen at, and what the other answers would have led to.

### The one command to run right now

```
python -m src.fixtures.generate                              # synthetic dataset, ~10s
python -m src.agent.run --data data_fixture --out cases_fixture
python -m src.io.validate cases_fixture --data data_fixture  # "All invariants pass."
python analysis/rubric_on_pack.py                            # the 20 REAL cases, cheap features
python -m pytest tests/ -q                                   # 231 passed, 36 skipped
```

That is the P2 milestone from section 6 - **20 valid answer files with zero external
dependencies** - reached on synthetic data. When the real dataset lands, the same three commands
produce the real twenty.

### When the dataset and keys land

```
python analysis/profile_dataset.py    # regenerates exam_triage.csv WITH the consistency columns
python -m src.features.build          # ~3 min, expect five [OK ] lines
python -m pytest tests/ -q            # the 36 skipped tests should now run
```

Note that nothing is committed — the working tree is dirty on purpose, so you can read the diff
before any of it becomes history.
