# Golden set: labelling notes

`data/golden/golden_pool.jsonl` lists the 150 cases to hand-label, in
labelling order. It holds case ids and strata only; `python -m src.label`
looks up each case's text in `data/sample/cases_clean.jsonl` as it shows it.
Labels are appended to `data/golden/golden_set.jsonl`, one line per case, as
each case is finished.

## How the pool was drawn

- Seed: 0 (`random.Random(0)`), used for every draw.
- Source: `data/sample/cases_clean.jsonl`, 39,582 cases.
- Size: 150, which is 120 stratified cases plus 30 hard cases.

### Exclusions

| rule | cases removed |
|---|---|
| one of the 46 cases quoted in `docs/taxonomy.md` | 46 |
| in the same thread as one of them (46 thread_ids) | 105 |
| in the same duplicate group as one of them (1 group, 1412659) | 1 |
| a near-duplicate of a kept case (one case per `dup_group_id`: the group's smallest `case_id`) | 229 |
| **eligible after exclusions** | **39,201** |

Asserted on the written pool: 150 distinct cases from 150 distinct threads,
none of the 46 example cases, their 46 thread_ids or their duplicate group,
and at most one case per `dup_group_id`.

### Hard cases: 30, chosen by rule, not by a model

The rules run in this order. Each picks 6 at random (seed 0), except the last,
and no case is picked twice. Candidate counts are eligible cases not already
picked by an earlier rule. "Content words" are tokens other than the `@USER`
and `URL` placeholders.

| stratum | rule | candidates | picked |
|---|---|---|---|
| hard:non_english | `lang` is a real language code other than `en`; `unknown` is excluded because it is mostly short English below the 0.90 confidence threshold | 1,041 | 6 |
| hard:screenshot_only | contains the `URL` placeholder and fewer than 6 content words | 547 | 6 |
| hard:charge_and_failure | mentions a charge or payment term and a bug or failure term (the `CHARGE` and `FAILURE` patterns in `src/golden.py`) | 126 | 6 |
| hard:short_reply | has prior turns and fewer than 5 content words | 1,310 | 6 |
| hard:longest | the 6 longest messages | 39,177 | 6 |

### Stratified cases: 120, by a phi3 intent ESTIMATE

- A 1,500-case pool was drawn from the eligible cases (seed 0, the first
  draw), and phi3 (`phi3:3.8b-mini-128k-instruct-q4_0`) estimated each case's
  intent from the compact classifier prompt.
- The distribution estimate uses N = 1,500 cached phi3 predictions: the full
  planned pool. The pass finished on 2026-09-15; that run took 60.0 min, 866
  of the predictions being new. The sampler now reads the predictions from the
  disk cache and makes no model call, and re-running it rebuilt a
  byte-identical pool.
- **The estimate is never a label.** It only decides which cases get sampled,
  and the labelling CLI never shows it. A case in stratum
  `estimated:billing_subscription` may well be labelled something else.
- Estimated intents over the 1,500 (not labels): content_unavailable 314,
  billing_subscription 226, feature_request 198, other_unclear 174,
  account_access 168, followup_diagnostic 156, playback_failure 116,
  library_playlists 81, chatter_thanks 67.
- Cases already picked as hard cases are skipped, and every intent must have
  at least 20 cases left (all do). The 120 are then drawn round-robin over the
  9 intents in taxonomy order. 120 / 9 leaves 3 over, and the round-robin
  gives them to the first three intents.

| stratum | cases |
|---|---|
| estimated:billing_subscription | 14 |
| estimated:account_access | 14 |
| estimated:playback_failure | 14 |
| estimated:library_playlists | 13 |
| estimated:content_unavailable | 13 |
| estimated:feature_request | 13 |
| estimated:followup_diagnostic | 13 |
| estimated:chatter_thanks | 13 |
| estimated:other_unclear | 13 |

### Labelling order

The 150 are shuffled (seed 0), so the hard cases are mixed in rather than
arriving as one block.

## What the labelling CLI shows and records

- **Shows:** the customer message and its prior turns. It never shows the
  brand's real reply, the language tag, the stratum or the phi3 estimate,
  because each would anchor the label.
- **Records per case:** intent (one of the 9), compromised (true/false),
  escalate (y/n), escalate_reason (free text, when escalating), difficulty
  (1 to 3), notes (free text) and labelled_at (UTC).

## Not distribution-matched

The golden set is deliberately not distribution-matched: the stratification
over-samples rare intents, and 30 hard cases are added on purpose. Every
metric must therefore be reported both stratified (on the set as labelled)
and reweighted to the estimated population intent shares (`report/REPORT.md`,
Section 3.1). A stratified figure alone is not a production estimate.
