# TRANSFER.md — moving machines and picking the work back up

Written 2026-09-21, moving off a Windows 11 box with 7.7 GB RAM / 39 GB free disk, which is
below TigerGraph's 8 GB minimum. Read this end to end once on the new machine before running
anything.

Order of reading afterwards: `PROJECT.md` (what and why) → `ARCHITECTURE.md` (how) →
`PLAN.md` (what's next) → `README.md` (the dataset itself).

---

## 1. What to move

| Item | Size | How |
|---|---|---|
| Git repo (32 files: all code, tests, docs) | <1 MB | `git push` / `git bundle`, or just copy the folder |
| `drive-download-20260919T105649Z-1-001/` | **704 MB** | Copy manually — gitignored, and it is the source dataset |
| `drive-download-...zip` | 65 MB | Optional; the extracted folder is what's used |
| `.env.local` | 4 KB | **Copy manually.** Gitignored by design — it holds live keys |
| `data/` (7 parquet files) | 39 MB | **Do not copy.** Rebuild in ~3 min, see step 4 |
| `analysis/exam_triage.csv` | 2 KB | In git already |

Total to physically move: **~745 MB** (or ~110 MB if you move the zip instead of the extracted
folder and re-extract).

The repo has **no remote**. Two options:

**Option A — push to GitHub now (recommended).** The submission requires a public repo anyway,
so this doubles as the transfer and gets that box ticked early:
```bash
gh repo create hhgoa-fraud-investigation --private --source=. --remote=origin
git push -u origin master
```
Then on the new machine `git clone`, and copy `drive-download-*/` and `.env.local` across by
USB/drive. **Verify `.env.local` did not get committed** — `git log --all --full-history -- .env.local`
must return nothing. It is in `.gitignore`, but check before making the repo public.

**Option B — copy the whole 807 MB folder.** Simpler, no GitHub yet. The `.git` directory comes
with it so history is preserved.

---

## 2. What the new machine actually needs

Be clear-eyed about which spec matters:

| Spec | Matters? | Why |
|---|---|---|
| **RAM** | **Yes, most** | TigerGraph CE minimum 8 GB, recommended 24 GB. It runs GSE, GPE, RESTPP, Kafka, Zookeeper, Nginx and the GUI regardless of data size. 16 GB is comfortable, 32 GB is generous |
| **Disk** | Yes | ~15 GB for TigerGraph + image, ~1 GB for the dataset, plus headroom |
| **CPU** | Somewhat | 8 cores is fine. The parquet build is single-threaded pandas and takes ~3 min |
| **GPU** | **Mostly no** — but see below | Nothing in the current design uses one |

### Where the GPU genuinely helps

The project does **no model training** — that path was tested and rejected (§6, finding ⑤).
TigerGraph does not use the GPU. So the GPU buys nothing for the core pipeline.

It does unlock one real thing, and it is the one you asked about earlier: **a local LLM for
development, so Anthropic credits are only spent on the final run.**

- With ~8 GB VRAM: a 7–8B instruct model (Llama 3.1 8B, Qwen 2.5 7B) via Ollama. Good enough to
  exercise the agent loop, prompt plumbing and JSON shape end to end.
- With ~24 GB VRAM: a 27–32B model, which starts being genuinely usable for the case summaries.
- Local embeddings (`sentence-transformers`, BGE/E5) instead of Voyage, if you want to drop that
  dependency too.

**Do not ship prose written by the local model.** Use it to prove the loop works, then run the
final 20 cases on Claude — that run costs about **$2.50**, so your $50 covers roughly 20 full
attempts. The local model is for the hundred iterations before that, not the deliverable.

If the new laptop has no usable GPU, nothing is lost: the deterministic core needs no LLM, and
the final run is cheap.

---

## 3. Environment setup

```bash
# 1. Python 3.10+ (3.10.10 used so far)
python --version

# 2. Dependencies already verified working
pip install pandas pyarrow numpy scikit-learn pytest

# 3. Not yet installed anywhere - needed for the agent/graph phases
pip install langchain-anthropic langchain-mcp-adapters pyTigerGraph voyageai
#   langgraph, langchain, anthropic, pydantic, jsonschema were already present
```

Restore `.env.local` (copied in step 1). It contains:
- `VOYAGE_API_KEY` + `VOYAGE_BASE_URL=https://ai.mongodb.com` (key name "goa")
- `TG_SECRET` and `TG_WORKSPACE_ID`
- `ANTHROPIC_API_KEY=` — **deliberately left empty**, see §2

**Rotate both keys if this repo or transcript has been shared.** They were pasted in plain text.

---

## 4. Rebuild and verify — do this before any new work

```bash
python -m src.features.build      # ~3 min, writes data/*.parquet
python -m pytest tests/ -q        # expect: 66 passed
```

The build asserts its own exit gate. All five lines must read OK:

```
[OK ] customers        13,553
[OK ] cards            14,893
[OK ] txns            590,742
[OK ] identity        144,432
[OK ] closed_cases      5,565
```

If any differ, the load is wrong — stop and fix before continuing. Optional deeper check:

```bash
python analysis/profile_dataset.py   # reproduces every figure quoted in the docs
python -m src.scoring.diagnose       # reproduces the rejected-model finding
```

---

## 5. Exact state at handover

**Done — 2,834 lines of code, 66 tests green, 4 commits, Phase 0b gate passing.**

| Module | Status | What it gives you |
|---|---|---|
| `analysis/profile_dataset.py` | ✅ | Every documented number, reproducible |
| `src/features/` | ✅ | card identity, as-of baselines, ring filter, R5 detector, parquet pipeline |
| `src/policy/` | ✅ 35 tests | R1–R10, routing, the two policy-3a gates — pure functions |
| `src/io/answer.py` | ✅ | the only place answer JSON is constructed |
| `src/io/validate.py` | ✅ 17 tests | 12 invariants; ERROR blocks submission |
| `src/graph/interface.py` + `local.py` | ✅ 14 tests | ~14 typed queries, parquet backend, as-of enforced |
| `src/scoring/features.py` | ✅ | feature extraction, shared by training and inference |
| `src/scoring/diagnose.py` | ✅ | evidence for why no model is fitted |

**Not built yet:**

| Piece | Blocked on | Notes |
|---|---|---|
| Rubric scorer (`src/scoring/rubric.py`) | nothing | **Next task**, see §7 |
| Agent loop (`src/agent/`) | nothing to write it; a key to run it | Local LLM is fine for development |
| `src/graph/tigergraph.py` | TigerGraph endpoint | Interface and contract tests already exist |
| MCP wiring | TigerGraph running | Judged requirement |
| Backtest runner (`src/eval/`) | rubric scorer | Qualitative only — see finding ⑤ |
| The 20 answer files | the above | `cases/` does not exist yet |
| UI | everything above | Capped at 4 h, cut first |

---

## 6. Findings that constrain the design — do not re-derive these

Full detail in `PROJECT.md` §3. Condensed, because each one cost real time to establish:

**① `card_id` must be reconstructed.** `transactions.csv` has no card column. Use
`(customer_id, card1..card6)` — 0 of 1,956 reachable keys are ambiguous. The K-suffix is **not**
derivable (first-seen, last-seen, volume, min-ID and card1 orderings all fail, best 60% ≈ chance).
Anchor it from the case files. Single-card customers are always `-K1` (1,430/1,430), which lifts
coverage to 86%. Cards still unknown must never be cited by ID in an answer.

**② Shared device profiles are mostly noise.** 4,793 of 9,706 span >1 card; the largest is 1,023
cards of all-null fields. Use the calibrated filter: full profile + ≤10 global cards + ≥2 cards +
≤14-day span, which cuts Nov–Dec candidates 1,671 → 266. High-volume customers still generate
false rings (HHG-018 hit 9× on 7,091 transactions), so require a low-volume card on the profile.

**③ Card testing is nearly absent.** 28 cards book-wide under the literal R5 reading, and **no
exam case qualifies at its flagged moment** at any threshold. A `card_testing` verdict needs
extraordinary support.

**④ Amount percentile beats `risk_score` for triage.** Six exam cases sit at the top of their own
card's history. The score ranks HHG-013 (0.76, 15th percentile, $36) above HHG-006 (0.25, 97th
percentile, $482, new device). See `analysis/exam_triage.csv`.

**⑤ Closed cases cannot train a classifier.** All 900 cleared cases have `risk_score` ≥ 0.81;
below that the history is 0 legitimate / 4,038 fraud. A fitted model scores AUC 0.991 by learning
*low score ⟹ fraud* — the inverse of reality — and **17 of 20 exam cases sit below 0.81**. No model
is shipped. `fraud_probability` comes from a transparent rubric instead.

---

## 7. What to do next, in order

### Step 1 — Rubric scorer (`src/scoring/rubric.py`) · no credentials needed

Replaces the rejected classifier. A transparent weighted function producing `fraud_probability`
plus the named signals it rested on, so the number is citable as evidence.

Weights are grounded in measured distributions, **not** fitted to closed-case labels:

| Signal | Direction | Grounding |
|---|---|---|
| `amt_pctile_prior` ≥ 0.95 | strong ↑ | best cheap discriminator in the pack (finding ④) |
| `amt_pctile_prior` ≤ 0.25 | strong ↓ | low amount + high score is the classic false alarm |
| `new_region` | ↑ | only 1 of 20 exam cases; rare therefore informative |
| `new_product` | mild ↑ | only 1 of 20 |
| rare shared device, tight window | strong ↑ | after the §6② filter, and only then |
| proxy present (`id_23`) | ↑ | 3.8% of identity records — genuinely rare |
| `id_15 = New` | **barely moves it** | 43% of all records, 11 of 20 exam cases |
| `risk_score` | **barely moves it** | brief says input not verdict; finding ⑤ shows why |
| customer disputes | ↑ | but R7 exists — check recurring pattern first |
| prior confirmed fraud on the customer | mild ↑ | as-of only |

Acceptance criteria:
- ≥2 independent signals required before the probability leaves the 0.15–0.85 band
  (policy §6 — enforce in code, don't hope)
- Returns `(probability, [(signal, contribution), ...])` so `Assess Uncertainty` can log drivers
- Run across all 20: verdict split must not be 20-0, and must not contradict `exam_triage.csv`
  without a stated reason
- Unit tests in `tests/test_rubric.py`

### Step 2 — Deterministic end-to-end run · still no credentials

Wire rubric → `PolicyContext` → `policy.evaluate()` → `answer.Answer` → `cases/*.json`, with
templated prose. Then:

```bash
python -m src.io.validate cases/     # must print "All invariants pass"
```

**This is the milestone that matters**: 20 valid, submittable files with zero external
dependencies. Everything after it is improvement, not rescue.

### Step 3 — TigerGraph

1. Stand up Savanna (preferred) or Docker CE **4.2.5** — *not* 4.3.0-rc1; TigerVector shipped in
   4.2, and an RC is a preview build. With ≥16 GB RAM, local Docker is now reasonable.
2. Implement `src/graph/tigergraph.py` against the existing `GraphBackend` interface.
3. Run the contract test: both backends must return identical results for all ~14 queries.
4. Load the schema from `ARCHITECTURE.md` §3; build the `NEXT` chain at load time.
5. Wire TigerGraph MCP — a judged requirement, not optional.

### Step 4 — Agent loop (`src/agent/`)

Seven LangGraph nodes per `ARCHITECTURE.md` §6. Two things that are easy to get wrong and both
silently cost points:

- `next_best_actions.initial` is captured at the **first** pass through Take Action, `final` at
  the **last**. Make it a property of the graph topology, not something Explain reconstructs.
- `tool_calls` / `tokens` / `latency_s` are **required output fields**. Instrument at the wrapper
  on day one; retrofitting is miserable.

Develop against the local LLM, then switch `AGENT_MODEL` to `claude-sonnet-5` for the real run.

### Step 5 — Qualitative backtest

Replay closed cases with the as-of cutoff. Given finding ⑤, treat the numbers as **directional,
not a score**: the population is not comparable to the exam pack. Most useful on the
`risk_score` ≥ 0.81 stratum (n=1,527, 41% fraud), which is the only place both outcomes occur —
but note it covers just 3 of the 20 exam cases.

### Step 6 — Submission

`cases/` complete and validating · all 20 written to the graph (query them back; `written_to_graph`
must reflect a real write) · 3–5 min demo video built around three cases, one of which changed its
recommendation after evidence · blog post — the strongest material is the card reconstruction, the
1,671→266 ring filter, and finding ⑤ · social post tagging @TigerGraphDB.

---

## 8. First 20 minutes on the new machine

```bash
git clone <repo> && cd hhgoa-fraud-investigation
# copy drive-download-20260919T105649Z-1-001/ and .env.local into place
pip install pandas pyarrow numpy scikit-learn pytest
python -m src.features.build        # expect 5x [OK ]
python -m pytest tests/ -q          # expect 66 passed
python -m src.scoring.diagnose      # see finding ⑤ for yourself
```

Green on all four → you are exactly where this handover left off, and Step 1 in §7 is the next
thing to write.
