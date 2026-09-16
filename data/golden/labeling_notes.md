# Golden set: labelling notes

**The golden set is MODEL-LABELLED, not hand-labelled.** 147 of its 150 cases
carry phi3's intent plus rule-derived escalate and compromised fields
(`mode: "model"`); 3 were labelled by me before model labelling was adopted
(see "3 labels by me" below). 40 of the 147 were then labelled independently
by me, blind, as an audit. Agreement between my audit labels and the model
labels: TBD until the audit is done; `python -m eval.label_stats` then writes
it into the Human audit section at the end of this file.

The classifier under evaluation and these labels come from the same model
family (phi3) and the same taxonomy, so any agreement between them is partly
shared error rather than accuracy, and macro-F1 against the model labels
measures consistency with phi3, not correctness. For the phi3 systems it is
stronger than that: a model label is phi3's answer to the classifier's own
prompt, so on a model-labelled case their intent prediction is the label
itself. Intent metrics are therefore scored only against the 43 human
labels: the 40 audit labels plus the 3 labelled by me.

## Files

| file | holds |
|---|---|
| `golden_pool.jsonl` | the 150 case ids and their sampling strata, frozen |
| `golden_set.jsonl` | one label per case: intent, compromised, escalate, escalate_reason, difficulty, notes, labelled_at, mode |
| `audit_queue.jsonl` | the 40 case ids to audit, in audit order |
| `audit_labels.jsonl` | my blind audit labels (`audit_intent`), kept apart from `golden_set.jsonl` |

None of them holds message text: the CLI looks each case up in
`data/sample/cases_clean.jsonl` as it shows it.

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

### Stratified cases: 120, by a phi3 intent estimate

- A 1,500-case pool was drawn from the eligible cases (seed 0, the first
  draw), and phi3 (`phi3:3.8b-mini-128k-instruct-q4_0`) estimated each case's
  intent from the compact classifier prompt.
- The distribution estimate uses N = 1,500 cached phi3 predictions: the full
  planned pool. The pass finished on 2026-09-15; that run took 60.0 min, 866
  of the predictions being new. The sampler now reads the predictions from the
  disk cache and makes no model call, and re-running it rebuilt a
  byte-identical pool.
- The estimate decided which cases were sampled. For the 147 model-labelled
  cases, the same cached prediction later became the intent label.
- Estimated intents over the 1,500: content_unavailable 314,
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

### Pool order

The 150 are shuffled (seed 0), so the hard cases are mixed in rather than
arriving as one block.

## How the labels were made

### 3 labels by me

- Cases 664584 and 930008 (`mode: "manual"`), labelled with
  `python -m src.label` before model labelling was adopted.
- Case 2363279 (`mode: "assisted"`), labelled with the since-removed
  `--assist` mode: phi3 suggested feature_request and I overrode it to
  library_playlists. It counts as a human label; its row is kept as written.
- Correction applied on 2026-09-15: case 664584 had been recorded with
  compromised = y by mistake and is now n, and its leftover test note "hi"
  was cleared.

### 147 model labels

- Written by `python -m src.model_label`, which makes no model call.
- **intent:** the sampler's cached phi3 estimate (the compact classifier
  prompt).
- **compromised:** a keyword rule over the customer's own text (the message
  and the customer's earlier turns): hack, hijack, someone or somebody else,
  someone changed / logged / is using / was using, not my device or account,
  isn't mine, unknown device, without me knowing.
- **escalate**, following the guideline, with the first matching reason:
  compromised ("account compromised"), billing_subscription ("needs account
  data"), other_unclear ("cannot determine intent"), followup_diagnostic with
  no earlier turns ("cannot interpret without earlier turns"). Otherwise not
  escalated.
- **difficulty** is null and **notes** are empty.

### The 40-case audit

- Drawn from the 147 model-labelled case ids, sorted, with
  `random.Random(0)`.
- `python -m src.label audit` shows each case blind: the customer message and
  its earlier turns only, never the model's label, the stratum, the language
  tag or the brand's reply. It asks for the intent only and appends my answer
  to `audit_labels.jsonl`.
- `python -m eval.label_stats` then reports agreement with a Wilson 95%
  interval, and per-intent agreement for intents with at least 5 audited
  cases.

## Not distribution-matched

The golden set is deliberately not distribution-matched: the stratification
over-samples rare intents, and 30 hard cases are added on purpose. Every
metric must therefore be reported both stratified (on the set as labelled)
and reweighted to the estimated population intent shares (`report/REPORT.md`,
Section 3.1). A stratified figure alone is not a production estimate.

<!-- audit_stats: start -->
## Human audit: agreement with the model labels

- Audited: 40 of 40, labelled blind by me.
- **Agreement with the model labels: 25 of 40 (62.5%), Wilson 95% interval 47.0% to 75.8%.**

Per intent, by the model's label; agreement is reported only where at least 5 audited cases carry that label.

| model label | audited | agreed | agreement |
|---|---|---|---|
| billing_subscription | 4 | 4 | not reported (support below 5) |
| account_access | 3 | 1 | not reported (support below 5) |
| playback_failure | 6 | 3 | 50.0% |
| library_playlists | 1 | 0 | not reported (support below 5) |
| content_unavailable | 5 | 3 | 60.0% |
| feature_request | 7 | 6 | 85.7% |
| followup_diagnostic | 5 | 3 | 60.0% |
| chatter_thanks | 3 | 3 | not reported (support below 5) |
| other_unclear | 6 | 2 | 33.3% |
<!-- audit_stats: end -->
