# DEMO.md: what we show, in what order, and why

The plan behind the 3-minute video. The word-for-word script, broken into recordable clips, is in
[`SCRIPT.md`](SCRIPT.md).

## The story in one sentence

> **The graph gathers the evidence, the rules make the decision, the model explains it, and all of
> it is fast, cheap and auditable.**

Every shot should prove one piece of that sentence. If a shot doesn't, cut it.

## The five things a judge must walk away with

| # | Claim | The shot that proves it |
|---|---|---|
| 1 | TigerGraph does the investigating | Terminal: `python -m src.graph.deploy --check` shows the full book live on Savanna. The graph panel in the console. `CASE-HHG-011` in Savanna's graph explorer |
| 2 | It knows when to ask, and asking changes the outcome | HHG-019: 0.81 *uncertain* → asks for step-up → 0.97 *fraud* → adds `FILE_REPORT (L2)`. The overview chart shows the same movement for all four cases that asked |
| 3 | It's fast and costs nothing to decide | The **Run it live now** button (tens of ms). Overview tiles: **$0.00**, 215 queries. The live monitor streaming |
| 4 | Humans stay in control | `L1`/`L2` routes on every action, the Approve and Reject buttons, the "Waiting on a human" queue |
| 5 | It's honest | The copilot's **Challenge it** on HHG-010, which found a real bug we then fixed. `SIMULATED` labels on assumed responses |

## Pre-flight checklist (do this 10 minutes before recording)

1. **Wake TigerGraph.** Savanna → your workspace → **Resume** (it auto-pauses after 60 idle
   minutes). While you're there: **Settings → enable auto-start**.
2. **Check the graph:**
   `python -m src.graph.deploy --check`. You should see Transaction 590,742 and FraudCase 20.
3. **Start the console with the copilot live:**
   `python -m src.ui.app`. The startup output should say `copilot : yes, claude-opus-5`.
4. **Open the console with face-cam mode on:** http://127.0.0.1:8000/?facecam#/overview.
   `?facecam` moves the copilot panel to the **bottom-left**, so your camera overlay in the
   bottom-right never covers it.
5. **Warm the caches** so nothing lags on camera:
   - Open HHG-011 and click **⚡ Run it live now** once.
   - Open the **Live monitor**, press start, then stop.
   - Open HHG-010, open the copilot and click **Challenge it** once. The answer streams in about
     6–12 s, and the second time it's cached and faster.
6. **Browser zoom at 110–125%**, so text is readable in a 1080p recording. Use light theme
   (brighter on video). Close other tabs.
7. **Tabs to have open, in this order:**
   1. Console, overview (`/?facecam#/overview`)
   2. GitHub repo, `cases/` folder (https://github.com/UJESH2K/HHGOA/tree/main/cases)
   3. Savanna console → workspace → GraphStudio / Explore Graph
   4. Terminal with `deploy --check` output visible

## The 20 answers

Produced by `python -m src.agent.run`. They pass all answer-file invariants
(`python -m src.io.validate cases`), and every case is written to TigerGraph and read back.

**6 fraud · 9 uncertain · 5 legitimate · 3 reports · 4 asked for evidence (all 4 changed their
recommendation) · 10 have an action waiting on a human · $0.00 model spend.**

| Case | Trigger | Verdict | p | Pattern | Exposure | Asked | Final actions (route if not auto) | Report |
|---|---|---|---|---|---|---|---|---|
| HHG-001 | model score | legitimate | 0.04 | none | $0.00 | step_up_auth | CREATE_CASE, CLOSE_NO_FRAUD | - |
| HHG-002 | model score | uncertain | 0.76 | card not present fraud | $292.36 | - | VERIFY_WITH_CUSTOMER, CREATE_CASE | - |
| HHG-003 | customer dispute | uncertain | 0.35 | none | $49.00 | - | BLOCK_CARD (L1), CREATE_CASE, ESCALATE_TO_ANALYST | - |
| HHG-004 | customer dispute | fraud | 0.87 | card not present new device | $128.33 | - | BLOCK_CARD (L1), CREATE_CASE | - |
| HHG-005 | model score | legitimate | 0.07 | none | $0.00 | - | CLOSE_NO_FRAUD | - |
| HHG-006 | customer dispute | fraud | 0.90 | account takeover | $482.12 | - | BLOCK_CARD (L1), CREATE_CASE | - |
| HHG-007 | model score | uncertain | 0.51 | account takeover | $111.92 | - | STEP_UP_AUTH, VERIFY_WITH_CUSTOMER, CREATE_CASE | - |
| HHG-008 | customer dispute | uncertain | 0.55 | card not present fraud | $55.68 | - | STEP_UP_AUTH, VERIFY_WITH_CUSTOMER, BLOCK_CARD (L1), CREATE_CASE | - |
| HHG-009 | customer dispute | uncertain | 0.14 | none | $30.02 | - | BLOCK_CARD (L1), CREATE_CASE, ESCALATE_TO_ANALYST | - |
| HHG-010 | model score | fraud | 0.89 | card not present new device | $1,000.03 | - | VERIFY_WITH_CUSTOMER, MONITOR_CONNECTED_CARDS, CREATE_CASE, FILE_REPORT (L2) | yes |
| HHG-011 | customer dispute | fraud | 0.98 | undocumented | $131.30 | - | BLOCK_CARD (L1), MONITOR_CONNECTED_CARDS, CREATE_CASE, FILE_REPORT (L2), ESCALATE_TO_ANALYST | yes |
| HHG-012 | model score | uncertain | 0.26 | account takeover | $30.91 | - | MONITOR_CONNECTED_CARDS, CREATE_CASE, ESCALATE_TO_ANALYST | - |
| HHG-013 | model score | legitimate | 0.05 | none | $0.00 | step_up_auth | CREATE_CASE, CLOSE_NO_FRAUD | - |
| HHG-014 | analyst request | uncertain | 0.50 | card not present new device | $74.96 | - | STEP_UP_AUTH, VERIFY_WITH_CUSTOMER, MONITOR_CONNECTED_CARDS, CREATE_CASE | - |
| HHG-015 | model score | fraud | 0.90 | card not present new device | $599.94 | - | VERIFY_WITH_CUSTOMER, CREATE_CASE | - |
| HHG-016 | customer dispute | uncertain | 0.53 | card not present new device | $59.67 | - | STEP_UP_AUTH, VERIFY_WITH_CUSTOMER, BLOCK_CARD (L1), CREATE_CASE | - |
| HHG-017 | model score | legitimate | 0.04 | none | $0.00 | - | CLOSE_NO_FRAUD | - |
| HHG-018 | customer dispute | uncertain | 0.52 | account takeover | $39.08 | - | BLOCK_CARD (L1), MONITOR_CONNECTED_CARDS, CREATE_CASE, ESCALATE_TO_ANALYST | - |
| HHG-019 | model score | fraud | 0.97 | undocumented | $99.92 | step_up_auth | VERIFY_WITH_CUSTOMER, MONITOR_CONNECTED_CARDS, CREATE_CASE, FILE_REPORT (L2), ESCALATE_TO_ANALYST | yes |
| HHG-020 | model score | legitimate | 0.17 | none | $0.00 | step_up_auth | CREATE_CASE, ESCALATE_TO_ANALYST, CLOSE_NO_FRAUD | - |

### The cases to feature, and what each proves

| Case | Why it's on screen |
|---|---|
| **HHG-019** | **The evidence loop.** It starts at 0.81 but rests on one family of evidence, so it's held *uncertain*. The agent prices three questions and asks for step-up. The (simulated) challenge isn't completed, so it moves to 0.97 *fraud*, and only *then* adds the regulatory report (L2) and escalation |
| **HHG-001** | The same question moving the other way: 0.18 → step-up passes → 0.04, closed as legitimate. Asking isn't a trick for finding fraud |
| **HHG-011** | A **rare device shared across customers**: in the same window the disputed card was used on a rare device profile that 3 other customers' cards also used. It blocks (L1), monitors the connected cards and files (L2) |
| **HHG-010** | **$1,000.03**, just over the §3a threshold, so it files. It's also the case where the copilot's *Challenge it* caught a real error in our report narrative |
| **HHG-014** | The analyst's own question ("several cards share an unusual device"). Honest answer: the profile is on 52 cards book-wide, a common phone rather than a rare one, so it verifies and monitors the connected cards rather than calling fraud |
| **HHG-017** | A 0.57 model score that is simply a normal purchase for this card, closed with no friction to the customer |

## Taking it forward (for the last 15 seconds, and for Q&A in Goa)

1. **A window-scoped ring traversal in GSQL**, so the graph backend matches the parquet backend
   exactly on shared-device rings. This is the one known gap, tracked as a strict expected-failure
   test.
2. **TigerVector:** semantic policy retrieval and hybrid similar-case search. The upgrade is
   written in `vector_upgrade.gsql`.
3. **A backtest over the 5,565 closed cases** with a calibration curve, so every rubric weight has
   a measured justification.
4. **A learning loop:** the analyst's approvals and rejections flow back into the graph as labelled
   outcomes, and the next similar-case query reads them.
5. **Real channels:** step-up and customer verification through actual messaging, replacing the
   simulated responses.

## If something breaks while recording

| Symptom | Fix |
|---|---|
| `deploy --check` errors or hangs | The workspace is paused. Resume it in Savanna and wait about 1 minute |
| Pill in the top bar says "copilot off" | `ANTHROPIC_API_KEY` isn't in `.env`, or you started the server before adding it. Restart `python -m src.ui.app` |
| "Run it live now" is greyed out | The server can't see `data/`. Run it from the repo root |
| Anything else | The public build (Vercel / `docs/index.html`) shows the same cases, replays and recorded copilot answers with no server at all |
