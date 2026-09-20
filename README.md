# Agentic Fraud Investigation — Dataset Guide

Data reference for the TigerGraph × Hacker House Goa fraud investigation task (IEEE-CIS edition).
Every number below was measured from the files in `drive-download-20260919T105649Z-1-001/`, not copied from the brief.

The organizers' brief — task, fraud policy, answer JSON schema, the 20 cases — lives in
[`drive-download-20260919T105649Z-1-001/README.md`](drive-download-20260919T105649Z-1-001/README.md).
**That document is what you are graded against. This one is about the data itself.**

---

## Contents

| File | Rows | Cols | Size | What it is |
|---|---|---|---|---|
| `transactions.csv` | 590,742 | 397 | 675 MB | Card transactions, Jul 2 – Dec 31 2016. 393 original Vesta columns + 4 added |
| `identity.csv` | 144,432 | 41 | 25 MB | Device/connection records for online transactions only |
| `closed_cases_history.csv` | 5,565 | 15 | 2.6 MB | Finished investigations, Jul – Nov 2016. The only ground truth in the dataset |
| `case_pack.csv` | 20 | 8 | 3.5 KB | The exam: 20 alerts to investigate, Nov – Dec 2016 |
| `README.md` | — | — | 33 KB | Organizers' brief (task, policy, answer format) |

UTF-8, header row, amounts in USD. There is **no fraud label** in `transactions.csv` — the label was replaced by `risk_score`.

---

## Entity model at a glance

```
Customer (13,553)
   └── Card (14,893 distinct)          ← derived, see "The card_id problem"
         └── Transaction (590,742)      TransactionID, unique, no duplicates
               ├── identity record      144,432 — online only, join on TransactionID
               ├── BillingRegion        addr1, 332 distinct codes
               └── EmailDomain          P_emaildomain / R_emaildomain

ClosedCase (5,565) ──INVOLVES──> 14,955 distinct transactions
                   ──ON_CARD───> 1,913 card_ids across 1,892 customers
```

---

## `transactions.csv`

397 columns = 393 original Vesta columns + `customer_id`, `ts`, `channel`, `risk_score`.

### The four added columns

| Column | Measured facts |
|---|---|
| `customer_id` | `C00001`-style. 13,553 distinct. Median 4 txns per customer, mean 43.6, max 14,932 — extremely skewed |
| `ts` | `2016-07-02 00:02:21` → `2016-12-31 23:58:54`. Real timestamps, derived from `TransactionDT` |
| `channel` | `in_person` 439,670 (74.4%) / `online` 151,072 (25.6%) |
| `risk_score` | 0.01 – 0.99, mean 0.171, median 0.12. Only 16,871 rows (2.9%) score > 0.7; 1,765 (0.3%) score ≥ 0.9 |

`channel` is a deterministic restatement of `ProductCD`: **`W` ⇒ `in_person`, everything else ⇒ `online`.** Verified exactly — all 439,670 `W` rows are `in_person` and none of them has an identity record.

### Monthly volume

| Month | Txns | Mean risk | Txns > 0.7 |
|---|---|---|---|
| 2016-07 | 130,295 | 0.198 | 6,075 |
| 2016-08 | 93,786 | 0.164 | 2,160 |
| 2016-09 | 92,903 | 0.163 | 2,175 |
| 2016-10 | 100,420 | 0.162 | 2,252 |
| 2016-11 | 84,805 | 0.164 | 1,984 |
| 2016-12 | 88,533 | 0.166 | 2,225 |

**Exam window (Nov 1 – Dec 31): 173,338 transactions, 4,209 of them scoring > 0.7.** That is the search space for the optional autonomous-monitoring track.

### Amounts

`TransactionAmt`: min $0.27, median $68.86, mean $135.08, p75 $124.96, max $31,937.38.
Only **1,443 online transactions are under $5** — the card-testing signal (3+ tiny online auths in an hour) is rare enough to find by brute force.

### Categoricals worth loading first

| Column | Distribution |
|---|---|
| `ProductCD` | W 439,670 · C 68,721 · R 37,699 · H 33,024 · S 11,628 |
| `card4` (network) | visa 384,887 · mastercard 189,298 · amex 8,328 · discover 6,652 · null 1,577 |
| `card6` (type) | debit 440,091 · credit 149,035 · null 1,571 · "debit or credit" 30 · "charge card" 15 |
| `addr1` (billing region) | 332 distinct codes |
| `addr2` (country) | 87 = home country (520,643 rows). Next: 60 (3,087), 96 (642) |
| `P_emaildomain` | gmail 228,436 · yahoo 100,969 · **null 94,480** · hotmail 45,261 · anonymous.com 37,018 |
| `R_emaildomain` | null in 453,289 rows (77%) — present mostly on non-`W` products |

### The unnamed feature blocks

`C1`–`C14` (counts), `D1`–`D15` (day deltas), `M1`–`M9` (match flags), `V1`–`V339` (Vesta engineered features).
Vesta never published per-column definitions. They are heavily null and usable as signals, but any evidence
claim that names one must say it is an unnamed model feature — the brief scores honesty here.

---

## `identity.csv`

41 columns, 144,432 rows, joins to `transactions.csv` on `TransactionID`.

**Join integrity, verified:** all 144,432 identity rows match a transaction. 144,432 of the 151,072 online transactions have an identity record (95.6%) — **6,640 online transactions have none**. Zero `W`/in-person transactions have one.

| Field | Distribution |
|---|---|
| `DeviceType` | desktop 85,204 · mobile 55,801 · null 3,427 |
| `id_15` (device state) | Found 67,773 · **New 61,754** · Unknown 11,653 · null 3,252 |
| `id_23` (proxy) | null 139,144 · TRANSPARENT 3,492 · ANONYMOUS 1,185 · HIDDEN 611 |
| `DeviceInfo` | 1,786 distinct. Top: Windows (47,741), iOS Device (19,805), MacOS (12,579) |
| `id_30` / `id_31` / `id_33` | 75 OS values · 130 browsers · 260 screen resolutions |

`id_15 = New` covers **43% of all identity records**. On its own it is close to a coin flip — the brief's pattern 3 warns about this, and the data agrees.

### Device profiles are mostly generic — read this before building ring detection

Building a device profile as `DeviceInfo | OS | browser | screen` yields **9,706 distinct profiles**, of which **4,793 appear on more than one card** and 1,757 on more than five. The top profiles are not rings, they are defaults:

| Device profile | Cards | Customers |
|---|---|---|
| `? \| ? \| ? \| ?` (all four null) | 1,023 | 1,011 |
| `Windows \| Windows 10 \| chrome 63.0 \| 1920x1080` | 846 | 842 |
| `? \| ? \| mobile safari generic \| ?` | 719 | 697 |
| `Windows \| ? \| chrome 63.0 \| ?` | 634 | 627 |

In the Nov–Dec window alone, 1,671 profiles touch more than one customer. **"Shared device profile" is only evidence when the profile is specific and rare** (full string, no nulls, low global card count) and the sharing is tight in time. A rule that fires on any shared profile will flag half the book and score badly under policy R6.

---

## `closed_cases_history.csv` — the only ground truth

5,565 investigations, opened Jul 2 – **Nov 2**, closed Jul 4 – Nov 6, 2016. This is both your labelled history and the agent's starting case memory.

| Outcome | Count |
|---|---|
| `confirmed_fraud` | 4,665 (83.8%) |
| `cleared` | 900 (16.2%) |

| Pattern | Cases | Mean exposure | Median | Total |
|---|---|---|---|---|
| `card_not_present_fraud` | 1,404 | $162.63 | $100.02 | $228,332 |
| `account_takeover` | 1,205 | $752.65 | $252.40 | $906,945 |
| `card_not_present_new_device` | 1,076 | $324.79 | $178.43 | $349,469 |
| `out_of_region_use` | 955 | $582.96 | $234.93 | $556,730 |
| `card_testing` | **16** | $1,273.16 | $677.72 | $20,371 |
| `undocumented` | **9** | $1,171.27 | $1,871.13 | $10,541 |
| `none` (cleared) | 900 | $0 | $0 | $0 |

Two things to take from that table:

- **`card_testing` and `undocumented` are rare** (16 and 9 cases). Few-shot retrieval will rarely surface them, and the 9 `undocumented` analyst notes are the highest-value text in the dataset — read all nine by hand.
- `report_filed = Yes` on only **397 of 5,565** cases (7.1%). Most cases never became a report. The brief says the same; the history proves it.

`actions_taken` takes exactly three values, which is effectively a label for the policy decision:

| Actions | Count |
|---|---|
| `CREATE_CASE\|BLOCK_CARD` | 4,268 |
| `VERIFY_WITH_CUSTOMER\|CLOSE_NO_FRAUD` | 900 |
| `CREATE_CASE\|BLOCK_CARD\|FILE_REPORT` | 397 |

Other measured facts:
- `exposure_usd`: median $117.09, mean $372.40, max $35,031.56. Cleared cases are always $0.
- `n_txns`: median 1, mean 2.69, max 356. **Most confirmed fraud is a single transaction.**
- `connected_card_ids` is populated on only **4 of 5,565** rows — near-useless as a training signal for ring links; you have to find connections in the graph yourself.
- `first_fraud_txn_id` is null exactly on the 900 cleared cases; all 4,665 non-null values exist in `transactions.csv`.
- All **14,955 distinct transaction IDs** referenced across `txn_ids` exist in `transactions.csv`. No dangling references.
- Coverage: 1,892 customers, 1,913 card_ids — 14% of the customer base has prior case history.

---

## `case_pack.csv` — the 20 exam cases

Triggers: `risk_score` 11 · `customer_report` 8 · `analyst_request` 1. Opened Nov 12 – Dec 29, 2016 — entirely after the last closed case (Nov 2), so there is no leakage from history into the exam window.

All 20 `flagged_txn_id` values exist in `transactions.csv`. **16 of the 20 customers already have closed-case history** (C02354: 20 prior cases, C09933: 19, C13171: 19, C11923: 11) — retrieve it, it is free evidence.

| Case | Flagged txn | Amount | Channel | ProductCD | addr1 | risk_score | Customer txns | Prior cases |
|---|---|---|---|---|---|---|---|---|
| HHG-001 | 3514030 | $77.07 | in_person | W | 444 | 0.61 | 422 | 4 |
| HHG-002 | 3478782 | $292.36 | online | C | — | 0.79 | 44 | 1 |
| HHG-003 | 3530164 | $49.00 | in_person | W | 330 | 0.40 | 1,140 | 6 |
| HHG-004 | 3583227 | $128.33 | online | C | — | 0.34 | 216 | 4 |
| HHG-005 | 3523199 | $100.07 | online | R | 330 | 0.54 | 92 | 3 |
| HHG-006 | 3476682 | $482.12 | online | C | 264 | 0.25 | 261 | — |
| HHG-007 | 3514948 | $111.92 | in_person | W | 264 | 0.87 | 2,792 | 19 |
| HHG-008 | 3558054 | $55.68 | online | C | — | 0.38 | 928 | 19 |
| HHG-009 | 3581141 | $30.02 | online | S | 203 | 0.28 | 56 | — |
| HHG-010 | 3506725 | $1,000.03 | online | R | 469 | 0.90 | 36 | 1 |
| HHG-011 | 3583368 | $131.30 | online | C | — | 0.39 | 10,361 | 11 |
| HHG-012 | 3553342 | $30.91 | in_person | W | 494 | 0.55 | 991 | 2 |
| HHG-013 | 3526826 | $35.66 | online | C | — | 0.76 | 1,569 | 4 |
| HHG-014 | 3478561 | $74.96 | online | C | 191 | 0.05 | 85 | — |
| HHG-015 | 3464869 | $599.94 | online | R | 327 | 0.77 | 79 | 3 |
| HHG-016 | 3534820 | $59.67 | online | C | — | 0.37 | 61 | — |
| HHG-017 | 3450629 | $100.09 | online | R | 204 | 0.57 | 59 | 1 |
| HHG-018 | 3491361 | $39.08 | in_person | W | 126 | 0.48 | 7,091 | 20 |
| HHG-019 | 3503878 | $99.92 | online | R | 264 | 0.90 | 248 | 4 |
| HHG-020 | 3509359 | $125.08 | online | C | — | 0.52 | 109 | 2 |

Note the mismatch on the customer-report cases: HHG-006 is a $482 dispute on a transaction the model scored **0.25**, and the analyst-request case HHG-014 scored **0.05**. The `risk_score` column in `case_pack.csv` is only filled for risk-score triggers, but the underlying transaction always has one, and for customer reports it is usually low. Pull it from `transactions.csv` rather than treating a blank as "no signal".

Also note HHG-011 (C11923, 10,361 transactions) and HHG-018 (C02354, 7,091). Any "unusual for this cardholder" test has to survive customers with five-figure transaction histories as well as customers with 36.

---

## The `card_id` problem

`transactions.csv` **has no `card_id` column**, but `case_pack.csv` and `closed_cases_history.csv` both key on one (`C01234-K1`). You have to reconstruct it. Suffixes observed: K1 (2,980 cases), K2 (2,578), K3 (7).

**The reconstruction that works, verified:** treat `(customer_id, card1, card2, card3, card4, card5, card6)` as the card key.

- Yields **14,893 distinct cards** across 13,553 customers (mean 1.10 cards per customer, max 4).
- **5,529 of 5,565 closed cases** have every one of their transactions on a single card key (36 span more than one — those 36 are likely genuine multi-card episodes, worth a look).
- **Zero ambiguity:** of the 1,956 `(customer, card key)` pairs reachable from closed-case transactions, **not one maps to two different `card_id`s**. The key is sound.

What the key does *not* give you is the K-number: the suffix is not recoverable from the card columns, and it is not ordered by volume (of 61 customers with two resolvable cards, the higher-volume card was K2 in 43 cases and K1 in only 17). So:

- **Anchor, don't guess.** For each of the 20 exam cases, look up the flagged transaction, take its card key, and bind that key to the `card_id` the case pack gives you. The `card_txns` column in the table above is the result of exactly that — e.g. HHG-011's `C11923-K2` resolves to 10,332 of C11923's 10,361 transactions.
- The same trick recovers card_ids from history: each closed case hands you `card_id` + `txn_ids`, so the mapping propagates.
- 10 of the 20 exam customers hold exactly one card, so the binding is trivial for half the pack.

```python
import pandas as pd

CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]

def card_key(df):
    return df.customer_id + "::" + df[CARD_COLS].astype(str).agg("|".join, axis=1)

txns = pd.read_csv("drive-download-20260919T105649Z-1-001/transactions.csv", low_memory=False)
txns["card_key"] = card_key(txns)

pack = pd.read_csv("drive-download-20260919T105649Z-1-001/case_pack.csv")
flagged = txns.set_index("TransactionID").loc[pack.flagged_txn_id]
key_to_card_id = dict(zip(flagged.card_key.values, pack.card_id.values))  # card_key -> "C01234-K1"
txns["card_id"] = txns.card_key.map(key_to_card_id)
```

Do the same over `closed_cases_history.csv` (`card_id` + first id in `txn_ids`) to label another 1,913 cards before you load the graph.

---

## Loading it

`transactions.csv` is 675 MB and 397 columns. Two practical notes:

- Pass `low_memory=False` or explicit dtypes — the V-block is sparse and pandas will otherwise infer mixed types per chunk.
- For most graph work you need ~15 of the 397 columns. Project them out first:

```python
CORE = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain",
        "customer_id", "ts", "channel", "risk_score"]
txns = pd.read_csv("transactions.csv", usecols=CORE)   # ~90 MB in memory
```

Suggested vertices and edges are in the organizers' brief; the schema-relevant cardinalities are:

| Vertex | Count |
|---|---|
| `Customer` | 13,553 |
| `Card` | 14,893 |
| `Transaction` | 590,742 |
| `DeviceProfile` | 9,706 (fewer if you drop all-null profiles) |
| `BillingRegion` | 332 |
| `EmailDomain` | 59 purchaser / 60 recipient domains |
| `ClosedCase` | 5,565 |

A `Transaction -[NEXT]-> Transaction` chain ordered by `ts` within a card key is what makes the card-testing and burst patterns queryable in one hop; build it at load time.

---

## Data cautions, in priority order

1. **`risk_score` is an input, not a label.** Mean 0.171 across the book; only 2.9% of rows exceed 0.7. Six of the eight customer-report cases sit below 0.40, including a $482 dispute at 0.25. An agent anchored to the score will miss them.
2. **Shared device profiles are mostly noise.** 4,793 profiles appear on more than one card; the largest is 1,023 cards of null fields. Require a specific profile and a tight time window before invoking policy R6.
3. **`id_15 = New` is 43% of all identity records.** Not evidence on its own.
4. **`card_id` must be reconstructed** and the K-suffix must be anchored from the case pack or closed cases — see above.
5. **Half the exam is meant to be legitimate.** The closed history is 83.8% confirmed fraud, which is the opposite balance. Do not calibrate `fraud_probability` on the history's base rate; it is a filtered population of alerts that were already worth investigating.
6. **Nulls are everywhere and meaningful.** `P_emaildomain` is null on 94,480 rows and `R_emaildomain` on 77%. `addr1` is null on 65,739 rows overall, but that is almost entirely the `C` product — **94.8% of `C` transactions have no billing region**, against under 2% for every other product code. A null is not a zero: seven of the 20 exam cases have no `addr1` on the flagged transaction, which rules out the out-of-region pattern for them on that transaction alone.
7. **Multi-card customers exist but are rare** (mean 1.10 cards). Policy R10 (`BLOCK_ALL_CARDS`) needs two cards showing confirmed fraud — check the customer even has two before reaching for it.
8. **Do not use the public IEEE-CIS/Kaggle files.** IDs, times and amounts were transformed here; the brief calls recovering outcomes that way disqualification.

---

## Attribution

IEEE-CIS Fraud Detection dataset, Vesta Corporation, via the IEEE Computational Intelligence Society.
Customers, calendar, channel, risk scores, closed cases and the case pack were added by TigerGraph for the Hacker House Goa 2026 task. Data is anonymized by its publisher; no real people.
