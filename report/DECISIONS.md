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
