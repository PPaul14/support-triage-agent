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
