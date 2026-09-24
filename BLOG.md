# The graph gathers, the rules decide, the model explains: building an agentic fraud investigator on TigerGraph

*Tidewatch: our build for the TigerGraph × Hacker House Goa 2026 Agentic Fraud Investigation challenge.*
*Code: https://github.com/UJESH2K/HHGOA*

A fraud alert is a question with an expensive wrong answer in both directions. If you block a
real customer's card, you've failed them. If you wave through a stolen card, you've failed the
bank. The challenge handed us 590,742 card transactions, a fraud policy with ten rules, 5,565 closed
investigations and 20 open alerts. It asked for an agent that investigates each alert, decides what
to do next, knows when to ask for more evidence, and explains itself well enough for an analyst,
and eventually a regulator, to act on it.

This post covers what we built, how TigerGraph sits at the centre of it, and the numbers. It also
covers the two times our own tools caught us being wrong.

---

## 1. What we built

Tidewatch works a fraud alert the way a careful analyst would:

1. **Investigate.** It pulls the evidence out of a transaction graph on TigerGraph Savanna: the
   flagged transaction, the card's own history, the device it ran on, whether that device is
   shared with other customers' cards, card-testing bursts, recurring charges, and the customer's
   past cases along with the closest prior cases across the whole book.
2. **Assess.** It scores that evidence with a transparent log-odds rubric, so every point of
   probability can be traced to a named signal.
3. **Decide whether to ask.** It prices every question it could put to the customer or an analyst
   by how far the answer could move the recommended action, minus the cost of asking. It asks only
   when that net value is positive.
4. **Act within policy.** It recommends actions from the policy's vocabulary. Each action carries
   the approval route the policy assigns it: `auto`, `L1` or `L2`.
5. **Remember.** It writes the finished case back into the graph as a `FraudCase` vertex, then
   reads it back before the answer file is allowed to say it was written.

On top of that sit an **analyst console** and a **Claude copilot**. The console replays every
investigation step by step, shows the case's neighbourhood in the graph, and runs an autonomous
monitor over the book. The copilot explains and challenges the decision, but never makes it.

**The design choice everything else follows from: no language model sits in the decision loop.**
Every verdict, probability, action and route comes from graph queries, the rubric and the policy
engine. Every answer file can be reproduced to the byte and audited line by line. The model does
the thing a deterministic engine can't: it talks to the human.

---

## 2. Architecture

```
                      ┌──────────────────── TigerGraph Savanna (4.2.5) ────────────────────┐
  alert               │  Customer ─OWNS→ PaymentCard ─USED_IN→ Transaction ─ON_DEVICE→      │
  (model score,  ───► │  DeviceProfile   Transaction ─NEXT_TXN→ Transaction                  │
   dispute,           │  PaymentCard ─SHARES_DEVICE─ PaymentCard   ClosedCase ─INVOLVES→ …   │
   analyst)           │  FraudCase ─ABOUT/CASE_CARD/TRIGGERED_BY/SIMILAR_TO→ …               │
                      │                                                                      │
                      │  GSQL: get_transaction · card_baseline · device_for_transaction      │
                      │        shares_device_edges · card_testing_window · same_amount_priors│
                      │        prior_cases_for_customer · similar_closed_cases · read_case   │
                      └───────────────┬──────────────────────────────────────▲──────────────┘
                                      │ evidence (two concurrent waves)      │ write_case → read_case
                                      ▼                                      │
   ┌──────────────┐   ┌────────────────────────┐   ┌───────────────────┐   ┌┴──────────────────┐
   │ log-odds      │──►│ value-of-information   │──►│ Fraud Policy v1.0  │──►│ case file + SAR   │
   │ rubric        │   │ (which question, if    │   │ R1–R10, §3a gates, │   │ actions routed    │
   │ (families,    │◄──│  any, is worth asking) │   │ approval routes    │   │ auto / L1 / L2    │
   │  band 0.15–.85)│   └────────────────────────┘   └───────────────────┘   └─────────┬─────────┘
   └──────────────┘                                                                    │
                                                                                       ▼
                                             ┌─────────────────────────────────────────────────┐
                                             │ Analyst console: replay · graph · approvals ·    │
                                             │ live monitor · Claude copilot (reads, never      │
                                             │ decides; cached policy + case prefix)            │
                                             └─────────────────────────────────────────────────┘
```

The investigation runs in seven stages: open, investigate, assess, gather more evidence, take
action, explain, update memory. Each stage is metered: graph queries, model calls, tokens, dollars,
wall time. Each step also goes into a millisecond-stamped trace, which the console replays.

There are two interchangeable graph backends behind one interface: TigerGraph (GSQL over REST++)
and a parquet backend for offline runs and backtests. A **contract test suite** runs the same
queries against both and fails if they disagree. It caught six divergences during the build, and
one would have crashed at runtime: the two backends returned card-testing episodes in different
shapes.

---

## 3. How we used TigerGraph

### The schema

We modelled the domain the way the investigation traverses it. The vertex types are `Customer`,
`PaymentCard`, `Transaction`, `DeviceProfile`, `BillingRegion`, `EmailDomain`, `ClosedCase`,
`FraudCase` and `PolicyChunk`. The edges follow the questions an investigator asks:

- **`OWNS` and `USED_IN`:** whose card, and which transactions.
- **`NEXT_TXN`:** the card's timeline, as a linked list, so burst and velocity questions are walks
  rather than sorts.
- **`ON_DEVICE`:** which device profile ran the transaction, carrying `device_state` and
  `proxy_kind` as edge attributes.
- **`SHARES_DEVICE`:** card-to-card links through a common device, the backbone of ring detection.
- **`ABOUT`, `CASE_CARD`, `TRIGGERED_BY`, `SIMILAR_TO`:** our own cases, attached to the entities
  they're about so the next investigation can find them.

The dataset has no card identifier, so we first reconstructed cards from the six card fields per
customer, giving 14,893 cards across 13,553 customers.

### Loading the full book

All 590,742 transactions went in, along with 575,849 `NEXT_TXN` edges, 144,432 `ON_DEVICE` edges
and 128,852 `SHARES_DEVICE` edges. Every count matched the source exactly. The first attempt, a
single 14 MB POST, hit a 504 gateway timeout, so the loader now sends 50,000-row chunks. The full
load takes about eleven minutes.

### Queries in the loop

Each investigation makes 10 or 11 graph calls: up to nine reads (card-present transactions skip the device lookup), then a write and a read-back.

| Question the agent asks | GSQL query |
|---|---|
| What exactly happened? | `get_transaction`, `get_customer` |
| Is this normal *for this card*? | `card_baseline` (percentile of the amount in the card's own history, new region, new product, 1h/24h velocity) |
| What device, and is it new or proxied? | `device_for_transaction` |
| Is the device shared with other customers' cards inside the window? | `shares_device_edges` |
| Was the card tested first (R5)? | `card_testing_window` |
| Is this a subscription the cardholder forgot (R7)? | `same_amount_priors` |
| Has this customer been investigated before? | `prior_cases_for_customer` |
| What did similar past cases conclude? | `similar_closed_cases` |
| Did our write land? | `read_case` |

The ranking of similar cases deliberately runs outside the query. The channel, pattern and
exposure filters are *relaxable*: a loosely filtered set of five is better evidence than a
perfectly filtered set of one. When the query applied them itself, the graph backend returned zero
similar cases where the parquet backend returned three. The one filter that is never relaxed, the
as-of cut, stays in GSQL, because a case closed after the one being replayed would hand the agent
its own answer.

### Case memory

Every finished case becomes a `FraudCase` vertex with edges to its customer, card, triggering
transactions and the prior cases it drew on. `written_to_graph` in an answer file is never a
constant. It's set only after `read_case` finds the vertex. All 20 graded cases are in the graph.

### Making it fast on Savanna

The first TigerGraph run took about **12 seconds per case**. Three changes brought it to about
**3 seconds**, with identical answers:

- **Concurrent waves.** Seven of the nine reads depend only on the alert, so they go out together.
  The other two need the channel and amount from the first wave.
- **One write, not eight.** The case vertex and all its edges now go in a single `upsertData` call:
  1.6 s → 0.23 s.
- **An immutable pool fetched once.** The closed-case history never changes, so it's read once per
  process and cut by date locally: 1.4–1.9 s per case → 9 ms.

On the in-memory parquet backend, the same investigation takes a **median of 40 ms**.

---

## 4. How the agent reasons

**The rubric.** Probability starts from a prior of 0.30 and moves in log-odds, one named
contribution per signal. Signals are grouped into *families*: amount behaviour, geography, device,
shared origin, customer response, and so on. One family on its own can't push a case out of the
0.15–0.85 band, however strong it looks. The bank's own risk score is deliberately capped just below
the materiality floor, so it's a reason to look but never an independent signal. The policy says
the same thing: "a score is a reason to look, never a verdict".

**Asking for evidence.** Before asking anything, the agent prices every option: step-up
authentication (friction 0.05), a call to validate with the customer (0.12), or a request to an
analyst (0.20). For each one it simulates the likely answers, re-runs the rubric and the policy on
each, and measures how far the *recommended action* would move. It asks only when expected movement
minus friction clears 0.05. Four of the twenty cases asked a question, and in all four the
recommendation changed. The dataset contains no real customer replies, so every response is
simulated and labelled `SIMULATED` wherever it appears.

**The policy engine.** Rules R1–R10 are encoded as functions that each cite their own rule. §3a,
"a case is not a report", is enforced as a gate. A regulatory filing needs a fraud verdict *and* an
aggravating condition: exposure over $1,000, a rare shared device, or an undocumented pattern.
Across 5,565 closed cases, only 7.1% ended in a report.

---

## 5. Where Claude fits

The copilot is `claude-opus-5` at low effort, called through the Anthropic Python SDK. It reads the
answer file, the diagnostics (every rubric contribution, every priced question, the decision log)
and the Fraud Policy. It cites evidence as `E3` and rules as `R6`. It may disagree out loud, but
changing the decision is a human's job through the approval route. The policy and the case form a
cached prefix, so a follow-up question costs about **$0.02**.

It earned its place. Asked to "challenge the decision" on HHG-010, it pointed out that the
shared-device evidence came from a *different device* than the flagged transaction ran on. It also
noticed that the draft suspicious-activity report named the wrong device as the one the
transaction "ran on". It was right. The ring query looks at every device the card used in the
window, and the report builder was picking the first profile from an alphabetically sorted list.
We fixed it, and the evidence text now says when the linked device isn't the transaction's own.
That's the division of labour we wanted: deterministic decisions, with a model that reads them
critically.

---

## 6. Results

| | |
|---|---|
| Verdicts across the 20 cases | 6 fraud · 9 uncertain · 5 legitimate |
| Regulatory reports | 3 of 20 (HHG-010 over $1,000; HHG-011 and HHG-019 on a rare device shared across customers) |
| Asked for evidence | 4 cases; the recommendation changed in all 4 |
| Actions needing a human | 10 cases have an `L1`/`L2` action awaiting sign-off |
| Model spend on decisions | **$0.00** (0 model calls) |
| Graph queries | 215 across the pack; 85% of evidence items come straight from the graph |
| Median investigation | 40 ms (parquet) · ~3 s (Savanna) |
| Validation | 20/20 pass the answer-file invariants; 276 tests pass |

**The autonomous monitor** watches the 4,462 highest-risk authorisations in the book, those the
bank's model scores at 0.85 or above, excluding the 20 graded ones. It investigates each end to end,
lets actions routed `auto` run (simulated, as the brief permits), and queues everything else for a
person. On a 300-alert sample it takes a median of 53 ms per alert and files reports on 12% of them.
That's more than the closed-case rate of 7.1%, which fits: these are the riskiest 2% of
transactions.

### The calibration story

The monitor did more than demo well: it found a bug. Early on, it labelled **90 of 300 unseen
alerts** as an `undocumented` typology. The closed-case history has *nine* such cases in 5,565.
We had loosened ring detection to catch a 52-card device profile named in one of the 20 cases, and
that loosened link was quietly qualifying for "new typology", which carries a mandatory report.
The fix was to require a *rare* device link. On the same alerts, `undocumented` dropped from 90 to
10 and reports from 33% to 12%.

The same review caught two more issues, both visible only when you read the answers against the
data:

- **An online purchase labelled `out_of_region_use`.** All 955 closed cases with that pattern are
  in person.
- **A report filed on a case the rubric itself held `uncertain`.** "Strongly suspected" now means
  the engine reached `fraud`.

None of this would have surfaced if we'd only looked at the 20 graded cases.

---

## 7. Honest limits

- **Simulated customer responses.** The dataset has none, so the agent assumes the most likely
  answer and says so every time.
- **A GSQL ring gap.** The ring query on TigerGraph undercounts device profiles that were also
  active outside the investigation window, because a precomputed edge attribute spans each card
  pair's whole history. The graded answers use the parquet computation. The gap is tracked as a
  strict expected-failure test, so a fix can't go unnoticed.
- **Vector search isn't live yet.** The TigerVector upgrade, semantic policy search and hybrid
  similar-case retrieval, is written in `vector_upgrade.gsql` but wasn't deployed for the graded run.
- **Nothing real is actioned.** Blocks and reports are recorded, never executed. What *is* real is
  which actions the policy allowed to run without a human.

## 8. What's next

- A window-scoped ring traversal over `ON_DEVICE`, so the graph backend matches parquet exactly.
- TigerVector for policy retrieval and hybrid case similarity.
- Replaying the 5,565 closed cases as a backtest, with a calibration curve.
- Letting the analyst's approvals flow back into the graph as labelled outcomes.

---

*Built at Hacker House Goa. TigerGraph Savanna for the graph, Python for the agent, Claude for the
copilot. Everything in this post is reproducible from the repository.*
