# Decisions

Design decisions, each with the alternative that was rejected. Numbers were
measured on the full `data/raw/twcs.csv` (2,811,774 rows) on 2026-09-11: the
`python -m src.ingest` summary plus one-off checks run during development.

## Ingest (`src/ingest.py`)

- **Brand = SpotifyCares.** 31.6% of its replies (13,124 of 41,497 cases) send
  the customer to DM, which is low enough that grounded replies are possible.
  - The DM share counts replies containing "DM" (exact case, also "DMs",
    "DM'd") or "direct message" (any case). It does not see deflection to
    phone, chat or links.

- **Only the earliest brand reply becomes `brand_reply`**, so answers split
  across several tweets are truncated. Known limitation, not fixed.
  - 1,466 of 41,497 cases (3.5%) have more than one direct brand reply. In
    1,299 of them every reply lands within 2 minutes, which suggests one
    answer split across tweets. None use "1/2"-style markers.
  - A further 103 cases have a first reply that the brand continued by
    replying to its own tweet; the continuation is cut as well.
  - Rejected: joining consecutive brand replies. The cost is small, and
    joining risks merging unrelated replies: in 167 of the 1,466 the replies
    are more than 2 minutes apart.

- **Threads with a missing parent are dropped**: 69 threads, 88 cases
  (133 brand tweets), so every retained case has its full context.
  - Rejected: keeping them with `prior_turns` starting at the earliest tweet
    in the file. Those cases would look complete while missing their opening.

- **Forks: `prior_turns` is the chain from the thread root to this message
  only**; sibling branches are invisible. 324 threads have cases on more than
  one branch.
  - Rejected: every earlier tweet in the thread, across branches, in time
    order. More context, but it mixes parallel conversations, including other
    customers' messages, into one.

## Clean (`src/clean.py`)

- **Near-duplicates are flagged, not removed.** Every case keeps its row and
  gets a `dup_group_id` (MinHash over character 5-grams of `customer_text`
  with `@USER` removed, Jaccard >= 0.85). Repetition is real traffic: 137
  groups cover 368 cases, and the largest is a 10-tweet
  `#SupportTwoFactorAuth` campaign.
  - The flag will be consumed in three places: excluding same-group
    precedents from retrieval (leak prevention), capping the golden set at
    one case per group, and reporting metrics both per case and per group.
  - Rejected: deleting near-duplicates. That distorts the traffic mix the
    triage decisions are measured on, and it cannot be undone downstream.

## Taxonomy (`src/taxonomy.py`)

- **`@USER` with punctuation attached is left in the clustering text** (for
  example `@USER,`, `@USER's`, `@USER...`). `src/taxonomy.py` strips `@USER`
  only as a whole token, so 180 of the 4,000 sampled messages still contain
  one, and "user" appears among the distinctive terms of clusters 2, 3 and 11.
  Known artifact, left unfixed.
  - Rejected: stripping every `@USER` form. That changes the embeddings and
    therefore the clusters the 8-intent taxonomy was merged from.

## Evaluation scope (decided 2026-09-12, before the harness)

Timing figures come from `python -m eval.latency` on 2026-09-12: 10 uncached
calls per model, with placeholder draft and judge prompts.

- **The golden set is 150 cases, not 200**: 120 stratified by intent (about 13
  per intent) plus 30 hard cases. 150 is the brief's floor, and it cuts every
  downstream stage by 25%.
  - The sampling rules carry over: at most one case per near-duplicate group,
    and none of the 46 cases quoted in `docs/taxonomy.md`, their threads or
    their duplicate groups.
  - Rejected: 200 cases.

- **B0 and B1 make zero LLM calls by construction.** B0 emits a constant
  reply; B1 is TF-IDF plus a nearest-neighbour copy of a past brand reply.
  Only the no-RAG ablation and the full system call llama3.

- **B0 is judged once per intent (9 calls) and the score reused**, because its
  reply is the same sentence every time.
  - This assumes the constant reply scores the same on every case of an
    intent: checks that depend on the case, such as whether the reply
    addresses the issue, are taken from one case per intent.
  - Rejected: judging the same sentence 150 times.

- **qwen2.5:7b judges everything; mistral:7b judges a 40-case sample only**,
  to measure cross-judge agreement. Judging was about two-thirds of the timed
  compute (13.6 of 20.3 h), and agreement between the two model families can
  be measured on a sample.
  - Rejected: both judges on every case.

- **num_predict is 80 for drafts and 150 for judges, and the rubric quotes
  evidence only for "no" answers.**
  - Drafts: the 4 genuine timing drafts were 39 to 48 tokens (144 to 204
    characters, 3.7 to 4.2 characters per token, so 280 characters is about 70
    tokens). The other 6 hit the 120 cap because llama3 copied the taxonomy
    block out of the prompt instead of replying (six identical 560-character
    outputs), not because replies ran long. The harness's draft prompt has to
    fix that.
  - Judges: at 200 tokens with evidence for every check, qwen hit the cap on 1
    of 10 calls and mistral on 2, with 1 judgement still unusable after
    repair. The 150 cap relies on the shorter "no"-only evidence, which has
    not been timed.
  - Rejected: 120 for drafts and 200 for judges.

  - **Revised on 2026-09-16: the judge cap is 200**, the value rejected above.
    Making the quote mandatory for every failing answer (Pipeline, below) made
    the replies longer: 1 of 8 smoke judgements ran past 150 tokens and was
    unusable even after its repair, and the usable ones reached 142. Drafts
    stay at 80.

- **Target: about 5.5 h for the full run, down from the 20.3 h baseline. Not
  yet measured.** Applying the measured per-call medians to the new call
  counts gives about 6.7 to 7.7 h, before the shorter judge outputs are
  counted; the range depends on whether mistral's 40-case sample means 40
  judgements or 40 per judged system. Re-time with the final prompts before
  the overnight run.
  - That arithmetic assumes one phi3 call per case for each LLM system (300
    calls at 4.4 s), 300 llama3 drafts at 16.7 s (the median of the 4 genuine
    drafts), 459 qwen judgements at 34.5 s (3 judged systems x 150, plus B0's
    9) and mistral at 46.7 s per call.

## Evaluation metrics (decided 2026-09-13, before any system is scored)

- **Headline metric: auto-handle rate at a fixed harmful-auto-reply budget.**
  A harmful auto-reply is a case labelled escalate that the system
  auto-handled. It is SEVERE when the true intent is billing_subscription or
  the case involves a compromised account, and ORDINARY otherwise. Hiver sells
  a shared-inbox helpdesk, and deflection at a stated safety budget is the
  number that maps to their product. The budget was then fixed at zero SEVERE
  and at most 5% ORDINARY (Evaluation method, below), before any system ran.
  - Rejected: a single escalation F1. Escalation is a threshold decision with
    asymmetric costs, and F1 weighs a harmful auto-reply the same as an
    unnecessary escalation.

- **Reply quality is judged with five binary checks, not a 1-5 scale:**
  contains_unsupported_specific, advances_resolution,
  addresses_stated_problem, tone_appropriate and would_send_unedited (the
  headline quality metric). Each check can be validated on its own with
  Cohen's kappa against hand scores.
  - Evidence is quoted for the answer that fails the reply: the unsupported
    span when contains_unsupported_specific is yes, and the reason when any
    other check is no. This restates the earlier "evidence only for no
    answers" rule, which assumed every check fails on no.
  - Rationale: a 7B-class judge tends to bunch holistic ratings together,
    which makes agreement on them meaningless. That is the expectation behind
    the choice; it has not been measured in this project.
  - Rejected: a holistic 1-5 quality score.

## Evaluation method (decided 2026-09-13, before any system is scored)

- **Harmful auto-replies are divided by all golden cases**: a per-ticket rate
  on the same denominator as the auto-handle rate.
  - Rejected: dividing by escalate cases only, which would restate 1 - recall.

- **The safety budget is pre-registered**, fixed before any system is scored
  so the headline cannot be tuned after the fact. SEVERE harmful auto-replies:
  zero tolerance; layers 1 and 2 of the escalation stack are deterministic and
  exist to make that achievable by construction. ORDINARY: at most 5% of all
  golden cases, which is at most 7 of 150. At n = 150 one case is 0.67
  percentage points, so the budget is coarse and the operating point is
  chosen at one-case resolution. The headline is the auto-handle rate at the
  grid point with the highest auto-handle rate that stays within the budget.
  - Rejected: choosing the budget after seeing the operating curve.

- **Systems are compared with a paired bootstrap on the per-case
  difference**: 2,000 resamples with a fixed seed, the same resampled cases
  scored for both systems, and an improvement stands only if the 95% interval
  of the difference excludes zero.
  - Rejected: comparing per-system confidence intervals. Overlapping
    intervals are not a test, and at n = 150 that distinction decides whether
    any claimed improvement stands.

- **Raw agreement is reported beside kappa for every judge check.** Kappa
  collapses when a check is nearly always "yes", as tone_appropriate almost
  certainly will be, even when the raters agree on almost every case. Kappa
  alone would understate agreement there; raw agreement alone would overstate
  it.
  - Rejected: kappa alone, with raw agreement only for would_send_unedited.

- **The 60 hand-scored replies are scored blind to the system that produced
  them.** The harness shuffles them and strips every system identifier before
  presenting them.

- **mistral's 40 (case, system) pairs are drawn from the 60 hand-scored
  replies**, so its agreement with the hand scores is computable. That fixes
  the sample at 40 pairs in total, which settles the range in the Evaluation
  scope timing entry at its 40-judgement end.

## Harness

- **The golden label records `compromised` as a boolean** next to the
  free-text `escalate_reason`, because the SEVERE harm rule needs it as data
  rather than prose. Free text goes in a separate `notes` field (renamed from
  `note`). Both changes were made before any case was labelled, so no label
  needed migrating.
  - Rejected: reading compromised accounts out of the free-text reason.

- **`src/llm.py` is 164 lines, an exception to the 150-line limit**, because
  `complete()` gained `bypass_cache` (154 lines) and then `logprobs` (see
  Pipeline). A temperature-0 self-consistency re-run
  through the cache returns the stored reply and would report 100% agreement
  by construction. The bypass neither reads nor writes the cache, so a re-run
  cannot replace the reply that later stages read. Keeping it in the one
  module that talks to Ollama matters more than four lines.
  - Rejected: a second Ollama client for re-runs, which would break the rule
    that every LLM call goes through `src/llm.py`.

- **`src/llm.py` retries read timeouts and 5xx replies, not only connection
  errors** (2026-09-17 and 2026-09-18). Two long runs died on transient Ollama
  failures that the retry loop could not see.
  - A call that hung past the 600 s read timeout killed the run at judgement 52
    of 459, after all four pipeline stages had finished: `requests.ReadTimeout`
    is not a `ConnectionError`, so the exception reached the top of the run.
    The handler now catches `requests.Timeout` as well.
  - The restarted run then died at judgement 308 of 459 on `HTTP 500: an error
    was encountered while running the model: unexpected EOF`, Ollama's model
    runner crashing mid-generation. `_chat` raised on every non-200, so a
    server-side crash was fatal where a dropped connection was not. Any 5xx is
    now retried with the same backoff; 4xx still raises, because a bad request
    will not fix itself.
  - Both fixes make a hung or crashed generation cost one backoff and a retry
    instead of the batch.
  - The disk cache made the restart cheap: the three pipeline stages replayed
    from disk in 0.7 minutes together, and only the unjudged replies were
    recomputed. That property is worth more than any single retry rule.
  - Rejected: raising the 600 s read timeout. That makes a genuinely stuck call
    take longer to fail without making it recoverable.

## Golden set (`src/golden.py`)

- **The stratification estimate is read from cached phi3 predictions only,
  and the sampler makes no model call.** N = 1,500 cached predictions, the
  full planned pool: the pass finished on 2026-09-15 (a 60.0 min run, 866 of
  the predictions being new) before the sampler was switched to reading the
  cache. At sampling time the shares only stratified; the same cached predictions
  later became the model labels (Golden set labels, below). Every
  intent must have at least 20 cached estimates left after the hard cases are
  removed, or the sampler stops and names the shortfall. The cache-only
  re-run rebuilt a byte-identical pool.
  - The earlier attempt was slow in wall-clock (645 logged calls over 4 h
    11 min, with a median of 4.0 s between calls) because of long pauses, not
    slow calls.
  - Rejected: calling phi3 from inside the sampler, which ties every sampling
    run to the model being available and to its pace.

## Golden set labels (decided 2026-09-15)

- **The golden set is model-labelled, not hand-labelled.** 147 of the 150
  cases carry phi3's cached intent estimate plus rule-derived escalate and
  compromised fields (`mode: "model"`); 3 were labelled by me. 40 of the 147
  are audited blind by me.
  - Rejected: labelling all 150 by hand.

- **Intent metrics are scored only against the 43 human labels** (the 40
  audit labels plus the 3 by me). A model label is phi3's answer to the
  classifier's own prompt, so if the classify stage uses that prompt, as
  `src/classify.py` is built for, its prediction on a model-labelled case is
  the label itself, and macro-F1 there is 1.0 by construction.
  - Rejected: scoring intent against all 150 labels.

- **escalate and compromised on the model labels follow the guideline's
  rules**: compromised by a keyword rule over the customer's own text, and
  escalate for a compromised account, billing_subscription, other_unclear, or
  followup_diagnostic with no earlier turns.
  - Rejected: escalate from the intent alone with compromised always false,
    which would miss compromised accounts, the SEVERE category.

- **The audit is 40 cases drawn with `random.Random(0)` from the sorted
  model-labelled case ids and labelled blind** (`python -m src.label audit`).
  The answers go to `audit_labels.jsonl`, apart from the model labels.

- **The --assist labelling mode was removed**; model labelling replaced it.
  It had been used for one case, 2363279 (phi3 suggested feature_request, I
  overrode it to library_playlists). That row is kept as written and counts
  as a human label.

- **Case 664584 was corrected**: compromised y to n, and a leftover test note
  cleared.

- **The first audit pass was invalid and was thrown away** (2026-09-16). Its
  40 answers stepped 1, 2, ... 9 straight through the keypad and repeated, all
  40 entered in 97 seconds, median gap 1.0 s, 16 of the 39 gaps under a second.
  That is keypad input from a CLI test, not labelling. Agreement with the model
  labels came out at 15%, close to the 11% that guessing over 9 intents gives,
  which is what raised the suspicion.
  - It was diagnosed with a shift test: each answer was joined to the queue
    case `shift` places away, for shifts -2 to +2. Shift 0 scored 15% and the
    four neighbours 8% to 11%, so the join was sound and the answers themselves
    were the problem; a real off-by-one would have shown a high-agreement
    neighbour. The stored model labels also matched phi3's cached estimates on
    all 40, which rules out the other side of the join.
  - The answers are kept as `artifacts/audit_labels_invalid.jsonl`, outside
    `data/golden/`, and the agreement section of `labeling_notes.md` is back to
    "Not audited yet".
  - **`python -m src.label audit` now guards against it** (`src/audit.py`): it
    refuses to write once 9 answers in a row step through the keypad, checked
    before each answer reaches the file, and it warns whenever an answer
    arrives in under 2 seconds. A file already on disk is held to a stricter
    limit of 5, because a refusal at the ninth answer leaves 8 behind and that
    part-cycle must not be extended quietly. Each answer's seconds are now
    stored beside it. Annotation tooling that cannot tell
    labelling from keypresses will hand you a number that reads like a finding.

## Judge validation (run 2026-09-19)

- **30 replies hand-scored, not the pre-registered 60.** The sample was halved
  to fit the time left; the cost is that every kappa in Section 3.4 rests on
  n = 30, and B0 was not drawn at all (12 no-RAG, 11 B1, 7 full), so B0's
  quality numbers have no human check.
  - Rejected: skipping the validation entirely, which is what the pre-registered
    60 would have meant in practice. A weak measurement of the judge is worth
    more than none, provided its weakness is stated.
- **The result was chance agreement on the headline check** — 53%, kappa 0.07
  on `would_send_unedited` — and it is reported as a finding rather than
  buried. Section 3.3 is caveated in place, not withdrawn: its marginal rates
  may still be roughly right (16 of my yeses against the judge's 14), while no
  per-reply verdict is trustworthy.
  - `tone_appropriate` collapsed to kappa 0.00 with 77% raw agreement, because
    the judge answered yes to all 30. Reporting both numbers was fixed in
    advance (Evaluation method, above) for exactly this case.
- **The blinding is partial, by construction.** The scorer sees the evidence
  the judge saw, and whether the agent was given past cases identifies the full
  system.
  - Rejected: hiding the precedents. `contains_unsupported_specific` is defined
    against that evidence, so hiding it would make the check unscoreable.
- **The hand-scoring CLI reuses the audit's guards, adapted to binary
  answers** (`eval/hand_score.py`). A keypad pattern — 12 answers all one key,
  or strictly alternating — is refused only when the last 3 replies were each
  scored in under 2 seconds. Either signal alone can be honest: five "no"
  answers in a row happen, and so does one quick obvious reply. Every reply's
  seconds are stored beside its answers.

## Pipeline (decided 2026-09-15, before any system is scored)

- **Stages run as batches, one model at a time**: phi3 classifies every case
  and then gives every layer-4 judgement, retrieval runs on the CPU, and
  llama3 drafts every reply last, so a run switches model once. `run(case)`
  exists for a single case, where the models necessarily alternate.
  - Rejected: taking each case through every stage in turn, which swaps
    models on every case.

- **Intent confidence is the probability phi3 gave the intent it wrote**:
  exp of the summed logprobs of the output tokens that spell it, from
  Ollama's token logprobs (`complete(logprobs=True)`). In a probe on Ollama
  0.34.0, a token the JSON schema forbids ranked first in `top_logprobs`, so
  these are the model's probabilities before the schema narrows the choice.
  `logprobs` joins the cache key only when requested, so every existing key,
  the sampler's estimates included, still matches.
  - The classify call keeps the sampler's exact prompt and settings, so on
    the 147 model-labelled cases the prediction is the label (Golden set
    labels, above).
  - Rejected: asking phi3 to state its confidence, and sampling several
    answers at temperature > 0, which breaks determinism and multiplies calls.

- **The precedent index leaves out every case that shares a thread_id or a
  dup_group_id with a golden case (486 of 39,582), and every case embedded
  within cosine 0.98 of one (10 more)**, leaving 39,086 precedents. B1 copies
  from the same list. The golden pool and label files must list the same 150
  cases or the build stops. `tests/test_index.py` asserts, for all 150 golden
  cases, that no retrieved precedent shares either id and that no similarity
  reaches 0.98, with its own 0.98 constant. It needs the gitignored data, so
  CI skips it.
  - The similarity rule was added after that test failed on its first run.
    Golden case 507403, a two-word thanks followed by two emoji, had five
    precedents at cosine 1.0000: other customers' same thanks with different
    emoji. MinHash put them at Jaccard 0.47 to 0.70, under 0.85, because in so
    short a string the emoji are much of the text, while MiniLM embeds them
    identically. No other golden case reached 0.95 against any precedent; the
    next highest was 0.94.
  - Rejected: filtering at query time only, which a caller could bypass.
  - Rejected: raising the test's threshold to let the case through.

- **Precedent intents come from a logistic regression on the MiniLM
  embeddings**, trained on phi3's cached estimates for the sampler's
  1,500-case pool minus any case sharing a golden thread or group: 1,370
  training cases. Retrieval filters on intent, and precedents have no labels.
  - Rejected: classifying all 39,096 precedents with phi3, about 48 h at the
    measured 4.4 s median.
  - Rejected: indexing only the 1,370 cases with an estimate.

- **The similarity floor (0.50) and the confidence threshold (0.50) are
  defaults, not tuned**, set before any golden retrieval or confidence was
  looked at. The operating curve (REPORT.md 3.2) sweeps both upward.
  Precedents below the floor are dropped one by one; when none is left the
  draft gets none and layer 3 escalates.

- **The no-RAG ablation retrieves nothing**, as REPORT.md Section 3 fixes: no
  precedents in the draft prompt and no similarity gate in layer 3.

- **The escalation ladder**: the first layer to fire decides.
  - L1: regex on the customer's own words (earlier customer turns and the
    message, never brand turns) for fraud, chargebacks, legal threats,
    account compromise, unauthorised charges, data deletion, self-harm, press
    and abuse. It needs no model output and nothing overrides it. Plain
    profanity is not abuse. The account-compromise rule is the regex that set
    the golden labels' `compromised` field, so on the model-labelled cases it
    agrees with them by construction.
  - L2: billing_subscription and other_unclear never auto-handle, and
    followup_diagnostic with no earlier turns escalates. That last rule is
    the guideline's split rule for the intent, added although the brief named
    only the first two; it is also the model-label rule, so it agrees with
    those labels by construction.
  - L3: intent confidence below the threshold, the best precedent below the
    floor, or a draft that failed its checks twice. The draft gate was added
    because an unusable draft must not be auto-sent.
  - L4: phi3 picks one of none, needs_account_data, needs_staff_action and
    unclear_request. reason_text is fixed per code, never the model's free
    text. phi3 is asked wherever L1 and L2 did not fire, including cases L3
    later escalates, so the threshold grid can be re-scored from the traces
    without a new model call.

- **Every case gets a draft, escalated or not**, because reply quality is
  judged on all 150; for an escalated case it is a suggestion for the human.

- **Draft checks**: a draft is rejected when it copies 8 words in a row from
  the prompt (the rules, the customer's words, the precedents' customer
  messages; a precedent's reply wording may be reused), contains a section
  marker, or names a refund, amount or timeline that no precedent reply
  contains. It is retried once with the reasons appended; a second failure
  keeps the text and escalates at layer 3. A draft over 280 characters, or
  cut off at num_predict 80, is trimmed to its last full sentence.

- **The precedent a draft used is named by word overlap**: the precedent
  whose reply shares the most words of 4+ letters with the draft, at least
  3, else none.
  - Rejected: asking llama3 to cite it. The brief wants the reply text only,
    and a JSON wrapper inside num_predict 80 risks truncated JSON.

- **B0 and B1 train on the same cached phi3 estimates and copy only from the
  index's leak-safe cases.** B0's intent is the majority estimate and its
  constant reply is a sentence I wrote. B1 is TF-IDF (unigrams and bigrams,
  min_df 2, sublinear tf) with logistic regression, a copy of the nearest
  case's reply by TF-IDF cosine with the leading @USER removed, and
  escalation on a fixed keyword list.
