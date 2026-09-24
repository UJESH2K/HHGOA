# SCRIPT.md: the 3-minute video, as four recordable clips

Record each clip on its own and stitch them together. Each one stands alone, so a fumble costs you
one clip, not the whole take.

**Setup for every clip**
- Screen recording at 1080p. Your face cam is a circle or rectangle in the **bottom-right**.
- The console always opened with `?facecam`, so the copilot sits bottom-*left*, clear of your
  camera: `http://127.0.0.1:8000/?facecam#/overview`.
- Browser zoom 110–125%, light theme.
- Do the pre-flight in [`DEMO.md`](DEMO.md) first: TigerGraph resumed, server started, caches
  warmed.
- **Pace:** about 140 words a minute. The **SAY** lines are written to that pace. If you run long,
  drop the lines marked *(optional)*.

| Clip | Length | Topic | Screens |
|---|---|---|---|
| 1 | 0:40 | The problem and the answer | console overview → GitHub `cases/` |
| 2 | 0:50 | TigerGraph does the investigating | terminal → Savanna → console case page |
| 3 | 0:50 | Judgement: when to ask, who decides | HHG-019 → copilot on HHG-010 |
| 4 | 0:40 | Autonomous, fast, and what's next | live monitor → overview |
| | **3:00** | | |

---

## Clip 1: The problem and the answer (0:00–0:40)

**Start on:** console → **Overview** tab, scrolled to the top.

| Time | DO | SAY |
|---|---|---|
| 0:00 | Face and the overview on screen. Don't touch anything | "A fraud alert has an expensive wrong answer in both directions. Block a real customer and you've failed them; wave through a stolen card and you've failed the bank." |
| 0:08 | Point the mouse at the headline, then the KPI tiles | "This is Tidewatch. It investigated all twenty alerts in the challenge in under a second, with zero dollars of model spend, and every decision traces back to a TigerGraph query or a rule in the fraud policy." |
| 0:20 | Scroll to the chart *"Where the 20 cases landed"*. Hover **HHG-019**'s row so the tooltip shows | "Six fraud, nine uncertain, five legitimate, with only three regulatory reports, because a case is not a report. The hollow dots are where a case stood *before* the agent asked for evidence." |
| 0:30 | Switch to the **GitHub tab**, `cases/` folder. Click **HHG-019.json** and scroll to `next_best_actions` | "All twenty answer files are in the repo, validated, and each one is written into the graph and read back before it's allowed to say so." |
| 0:40 | *cut* | |

---

## Clip 2: TigerGraph does the investigating (0:40–1:30)

**Start on:** the terminal, with `python -m src.graph.deploy --check` already run and its output on
screen.

| Time | DO | SAY |
|---|---|---|
| 0:40 | Terminal output. Run the mouse down the counts | "Everything runs on TigerGraph Savanna: the full book of five hundred and ninety thousand transactions, the card timelines, the devices, and a hundred and twenty-eight thousand shared-device edges between cards." |
| 0:50 | Switch to **Savanna → GraphStudio / Explore Graph**. Search the vertex **FraudCase `CASE-HHG-011`** and expand one hop | "Every investigation is a set of GSQL queries: the card's own baseline, the device, who else used that device, card-testing bursts, and the customer's history. When it finishes, it writes the case back as a vertex, so the next investigation can find it." |
| 1:03 | Switch to the console → **Cases** → **HHG-011**. Point at the graph panel on the right | "Here's the same case in our console: the customer, the card, the flagged transaction, and the other customers' cards linked through a rare shared device." |
| 1:12 | Click **▶ Replay** and let it run for about 5 seconds | "This is the investigation replayed step by step, with real timestamps: every query, every piece of evidence, the probability moving." |
| 1:20 | Click **⚡ Run it live now**. Point at the "Ran just now in … ms" line | "And this is it running live, right now: the whole investigation in tens of milliseconds." |
| 1:30 | *cut* | |

> If GraphStudio is slow to load, skip the 0:50 shot and stay on the terminal while you say that
> line. The case page's graph panel carries the point.

---

## Clip 3: Judgement: when to ask, and who decides (1:30–2:20)

**Start on:** console → **Cases** → **HHG-019**, scrolled to the top.

| Time | DO | SAY |
|---|---|---|
| 1:30 | Point at the verdict stamp, then scroll to **Next best action** (the before and after columns) | "The part we're proudest of is knowing when to ask. HHG-019 started at point eight one, but on a single family of evidence, so the rubric wouldn't call it." |
| 1:40 | Scroll to **"Every question it priced"**. Point at the ✓ on `step_up_auth` | "So it priced every question it could ask, by how far the answer would move the action minus the cost of asking, and chose step-up authentication. The response is simulated and labelled as such. Fraud moved to point nine seven, and only *then* did it add the regulatory report." |
| 1:55 | Point at a **`L2`** chip, then the **Approve** button | "Every action carries its approval route from the policy. Only the auto-routed ones run by themselves; the report waits for a human." |
| 2:03 | Click the case **HHG-010** in the left list. Click **Ask Claude about this case** (it opens bottom-left), then **Challenge it** | "Claude sits beside the analyst. It reads the case and the policy, and it can argue with the decision, but it never makes it." |
| 2:12 | Let the answer stream. Point at the cost line underneath | "It caught a real mistake in our report narrative, which we fixed. About two cents a question." |
| 2:20 | *cut* | |

---

## Clip 4: Autonomous, fast, and what's next (2:20–3:00)

**Start on:** console → **Live monitor**, not started yet, pace set to **brisk**.

| Time | DO | SAY |
|---|---|---|
| 2:20 | Click **▶ Start watching**. Let the feed fill | "For the autonomous track, the agent watches the stream: the riskiest four and a half thousand alerts in the book, each investigated end to end in about fifty milliseconds." |
| 2:30 | Point at the **Queued for a human** panel, then the KPI tiles | "What policy allows, it handles. What needs sign-off, it queues. Watching this stream is also how we caught our own over-eager rule and fixed it." |
| 2:40 | Click **Stop**. Switch to the **Overview** tab | "Next: a window-scoped ring query in GSQL, TigerVector for policy search, and a backtest over all five and a half thousand closed cases." |
| 2:50 | Look at the camera | "The graph gathers, the rules decide, the model explains. That's Tidewatch, built at Hacker House Goa on TigerGraph. Thanks for watching." |
| 3:00 | *end* | |

---

## Stitching checklist

- [ ] The four clips in order. Trim dead air at each start and end, and keep a 0.3 s crossfade at
      most.
- [ ] Total between **2:50 and 3:30**. The limit is 3–5 minutes, so anything up to 5:00 is legal,
      but a tight three minutes lands better.
- [ ] Face cam visible bottom-right in every clip and never covering the copilot (use `?facecam`).
- [ ] Optional lower-third captions at the start of each clip: *"TigerGraph Savanna · 590,742
      transactions"*, *"Evidence loop: HHG-019"*, *"Claude copilot: reads, never decides"*,
      *"Autonomous monitor"*.
- [ ] Upload unlisted to YouTube, or to Google Drive with *anyone with the link can view*.
      **Open the link in a private window** to check it plays before you paste it into the form.

## B-roll, if you want 20–30 seconds more

- The **How it works** tab: the architecture diagram and the five fact cards.
- Dark mode: the moon button, top-right.
- HHG-001 in the overview chart: the same question moving a case *down* to legitimate.
