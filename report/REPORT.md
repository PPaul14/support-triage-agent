# Report: an evaluated local triage agent for SpotifyCares

Status: complete, 2026-09-19. Every number was measured in this repository and
names its source; nothing is estimated or projected. Where something was not
run, it says so and says what it would take. The results come from the
150-case evaluation run of 2026-09-19 (`artifacts/eval/results.md`).

## 1. Problem framing

### What "good" means for SpotifyCares

SpotifyCares answers customers in public, on Twitter. Four things follow from
that channel, and together they decide what a triage agent is worth here.

- **A reply is published, not sent.** Everyone reading the thread sees it, and
  it stays there. A wrong public answer costs more than a slow one, which is
  why the headline in Section 3 is the auto-handle rate at a fixed
  harmful-auto-reply budget rather than an accuracy: the budget is the
  constraint, and deflection is the number that moves inside it.
- **Most of what the customer needs, the agent cannot see.** Charges, refunds,
  plan state and identity live behind account data a message-only agent has no
  access to. For those cases "good" is a fast, correct hand-off, not an answer.
- **A good reply either moves the case forward or moves it to the right
  place**: a concrete step the customer can take now, or a hand-off with the
  reason stated. A polite acknowledgement that does neither looks like service
  and resolves nothing. That is why `advances_resolution` is one of the five
  reply checks (Section 3.3), and why similarity to the brand's own reply is a
  diagnostic here and never a headline (finding 1.1).
- **What the agent actually buys is a person's attention.** Every case it
  handles safely is one nobody opens. That is the number a shared-inbox product
  is bought on, and it means nothing except next to the harm rate it was
  bought at.

What this project does not claim to know: SpotifyCares' own service targets,
staffing or response times. None of that is in the dataset, so "good" here is
defined from the channel and the data, not from their internal goals.

### Measured constraints

**1.1 A third of the brand's own replies send the customer to DM.** 31.6% of
SpotifyCares' replies (13,124 of 41,497 cases) mention "DM" or "direct
message". That is a floor on replies that move the conversation out of the
public thread: the check only sees those two phrases, so deflection to phone,
chat or links is invisible to it. Any metric that compares a drafted reply
with the brand's real reply inherits this base rate. Measured: B1, which copies
a real brand reply verbatim, scores highest of all four systems on ROUGE-L
(0.251) while scoring lowest on would_send_unedited (0.074). Similarity to the
reference inverts the ranking that matters (Section 3.6).

- Source: `python -m src.ingest` summary (commit cdd23c5); the matching rule is
  in `report/DECISIONS.md`, Ingest.

**1.2 Policy puts part of the traffic beyond any message-only agent.** In the
4,000-message clustering sample, the clusters merged into
billing_subscription hold 713 messages (17.8%). That intent always escalates:
resolving it needs account and payment data the agent has no access to. The
clusters merged into followup_diagnostic hold 478 (12.0%): replies whose
meaning lives in the earlier turns, which a message-only agent cannot act on.
Together that is 1,191 of 4,000 (29.8%): roughly 29% of this traffic is beyond
any message-only agent before a single model runs. The ceiling on the
auto-handle rate is set by the traffic and the escalation policy, not by the
modelling, and an auto-handle rate far above 70% on this mix would mean the
agent is answering cases the policy says it must not.
These are cluster shares, not labels: clusters are impure, the sample holds
one message per near-duplicate group, and the guideline moves cluster 1's
resolution confirmations to chatter_thanks. The labelled shares of the 150
golden cases are: followup_diagnostic 13.3%, billing_subscription 12.7%,
other_unclear 12.7%, playback_failure 11.3%, account_access 10.7%,
feature_request 10.0%, content_unavailable 10.0%, chatter_thanks 10.0%,
library_playlists 9.3%. That set is deliberately not distribution-matched
(`data/golden/labeling_notes.md`) and 147 of its labels are phi3's, so it
corrects the cluster shares only loosely. 48 of the 150 are labelled escalate,
which is the policy ceiling showing up in the sample.

- Source: `artifacts/clusters_summary.md` (commit 105a6b3); the handling
  policies are in `docs/taxonomy.md`.

### What I chose not to build

- **No multi-turn dialogue policy.** The unit here is one inbound message with
  the turns before it. Evaluating a policy across several of the agent's own
  turns needs live traffic or a simulated customer, and the dataset has
  neither: it holds the brand's real reply, never the agent's.
- **No fine-tuning.** The evaluation is the point of the project, and the
  obvious training target would undermine it: 31.6% of the brand's visible
  replies move the customer to DM (finding 1.1), so a model trained to imitate
  them would learn to deflect, and would score well on reference similarity for
  doing so. The machine is also CPU-only with 16 GB of RAM.
- **No tool execution.** The agent looks nothing up and changes nothing. That
  is the premise the escalation ladder exists to enforce: when a case needs
  account data, the correct output is a hand-off, not an attempt.
- **No live service.** A queue, rate limits and monitoring would add
  operational surface without changing a single measured number.
- **No sentiment feature for escalation, deliberately.** Anger and risk are
  different variables. An angry customer whose songs will not play is still
  auto-handleable; a calm customer whose account has been taken over is not.
  Escalating on tone would spend human time on volume rather than on need, and
  would bury the cases that matter under the ones that shout.

## 2. System

### The pipeline, as built

Nine stages, one module each. Every LLM call goes through `src/llm.py`, the
only module that talks to Ollama: a disk cache keyed on model, prompt and
parameters, one JSON repair retry, and a token and latency log.

| # | stage | module | model | output |
|---|---|---|---|---|
| 1 | ingest | `src/ingest.py` | none | threads rebuilt from `twcs.csv`: one case per inbound message, with its earlier turns and the brand's reply |
| 2 | clean | `src/clean.py` | none | normalised text, URLs and handles as placeholders, a language tag, near-duplicate groups (MinHash, Jaccard 0.85) |
| 3 | taxonomy | `src/taxonomy.py` | MiniLM | 25 KMeans clusters over 4,000 messages, merged by hand into the 9 intents of `docs/taxonomy.md` |
| 4 | classify | `src/classify.py` | phi3 3.8B | one intent, and the probability phi3 gave it, read from Ollama's token logprobs |
| 5 | index | `src/index.py` | MiniLM | 39,086 precedents, each with an embedding and an estimated intent; the leak exclusions are applied here |
| 6 | retrieve | `src/retrieve.py` | none | up to 5 precedents of the predicted intent above a 0.50 cosine floor, or none at all |
| 7 | draft | `src/generate.py` | llama3 8B | a reply under 280 characters, checked and retried once if it fails |
| 8 | escalate | `src/escalate.py` | phi3 3.8B (layer 4 only) | auto-handle or escalate, with a reason code and the layer that decided |
| 9 | run | `src/pipeline.py` | — | one trace per case: every intermediate value, prompt hash, token count and latency |

Stages run as whole batches, one model at a time: phi3 classifies all 150 cases
and gives every layer-4 judgement, then llama3 drafts all 150. Ollama holds one
model in RAM at a time on this machine, so going case by case would reload a
model twice per case.

**Retrieval is leak-safe by construction.** The index leaves out every case
sharing a `thread_id` or a `dup_group_id` with a golden case, and every case
embedded within 0.98 cosine of one: 486 plus 10 of 39,582 removed.
`tests/test_index.py` asserts, for all 150 golden cases, that no retrieved
precedent shares either id and that no similarity reaches 0.98. It failed on
its first run (Section 4, mode 3).

### The escalation ladder

Four layers, read in order. The first that fires decides, and no later layer
can overturn an earlier one.

| layer | what it is | fires on |
|---|---|---|
| L1 | regex over the customer's own words, consulted before any model output | fraud, chargebacks, legal threats, account compromise, unauthorised charges, data deletion, self-harm, press, abuse |
| L2 | the intent policy from `docs/taxonomy.md` | `billing_subscription` and `other_unclear` never auto-handle; `followup_diagnostic` with no earlier turns escalates |
| L3 | gates on the pipeline's own numbers | intent confidence below 0.50, best precedent below the 0.50 floor, or a draft that failed its checks twice |
| L4 | phi3, constrained to a fixed reason-code enum | `needs_account_data`, `needs_staff_action`, `unclear_request`; `none` means auto-handle |

L1 and L2 are deterministic, which is what makes the pre-registered
zero-SEVERE budget achievable by construction rather than by hope. L4 is asked
wherever L1 and L2 did not fire, including on cases L3 escalates, so the
threshold grid in Section 3.2 can be re-scored from the traces without another
model call.

**L1 and L2 share their rules with the labels they are scored against.** L1's
account-compromise regex is the rule that set the `compromised` field on the
147 model labels, and L2's `followup_diagnostic` rule is the one that set their
`escalate` field. On those cases the agreement is construction, not evidence.
Section 5 states what that costs; the blind audit covers intent only, so it
does not repair it.

**2.1 KMeans found axes other than topic.** Of the 25 clusters over 4,000
customer messages, clusters 1, 18 and 24 are conversational positions rather
than topics (replies reporting step results, device and version strings, DM
pointers), cluster 22 is a language (Indonesian and Tagalog), and cluster 11
is a modality (a link or screenshot carries the content). The taxonomy merges
these axes deliberately: the positions become followup_diagnostic, and the
language and the modality become other_unclear.

- Source: `src/taxonomy.py` and `artifacts/clusters_summary.md` (commit
  105a6b3); `docs/taxonomy.md`, "How this taxonomy was induced".

**2.2 A confident language tag was not a correct one.** langdetect tagged
plain English as "af" or "so" at confidence 1.0 (an example is recorded in the
`CleanCase` comment in `src/clean.py`). Adding a 0.90 confidence threshold
changed 2,511 labels, and 1,601 of those were messages previously tagged "en"
that became "unknown". "af" tags fell from 184 to 90, and the ones left still
include plain English at confidence 1.0. High confidence did not mean a
correct tag. The same question applies to the classifier's own stated
confidence, which has to be measured against the golden set rather than
assumed. **Not measured.** Layer 3 gates on that confidence at 0.50 and the
operating curve sweeps it, but no reliability curve or Brier score was computed,
so nothing here shows that a 0.6 means what a 0.9 means (Section 6, item 5).

- Source: `python -m src.clean` output before and after the switch to
  `detect_langs` (commit 07a864f), compared on 2026-09-11.

**2.3 The guideline written for people is too large to be the prompt.**
`docs/taxonomy.md` is 6,259 phi3 tokens (5,778 before its Summary lines were
added), and one call evaluating it took 178 s on CPU. The compact block
rendered from the same file is 921 tokens. With a 2,048-token context, Ollama
read only 1,026 tokens of the 6,259-token guideline and returned no error.
Whether the compact block classifies as well as the full guideline was **not
tested**: it needs a second classify pass at num_ctx 8,192 over the same cases,
about 11 minutes of phi3 at the measured 4.2 s per call, scored on the 43 human
labels.

- Source: measured through `src/llm.py` on 2026-09-12 (5,778 tokens) and
  2026-09-13 (6,259 tokens at num_ctx 8,192; 1,026 tokens at num_ctx 2,048);
  921 printed by `python -m src.classify` (commit 85419b3).

**2.4 Local inference budget.** Median seconds per call after the first, over
10 back-to-back uncached calls per model at num_ctx 4,096:

| stage | model | median | max | source |
|---|---|---|---|---|
| classify | phi3 3.8B | 4.4 s | 6.2 s | `python -m eval.latency`, 2026-09-12 (commit 85419b3) |
| draft | llama3 8B | 25.8 s | 32.0 s | `python -m eval.latency`, 2026-09-12 (commit 425ecd6) |
| judge | qwen2.5 7B | 34.5 s | 46.8 s | same run |
| judge | mistral 7B | 46.7 s | 87.9 s | same run |

Extrapolating these runs, a naive full run (200 cases x 4 systems classified
and drafted, 200 x 3 systems x 2 judges) comes to 20.3 h. The scoped run
recorded in `report/DECISIONS.md` (150 cases, LLM calls for two systems only,
B0 judged once per intent, mistral on a 40-case sample, output caps of 80 and
200 tokens) **now measures about
5.0 h of compute**: the no-RAG ablation 44.5 min and the full system 54.0 min
end to end (2026-09-16), the baselines 0.3 min, and 459 judgements at the
measured medians, about 3.3 h. Wall clock is not the right figure here: the run
was restarted three times and the machine slept overnight, so the sum of stage
medians is the honest number and the elapsed time is not.

What the timing numbers do not show:

- Ten calls per model, with placeholder draft and judge prompts.
- 6 of the 10 timing drafts were llama3 copying the taxonomy block out of the
  prompt instead of replying. The median over the 4 genuine drafts was 16.7 s,
  and the judge timings graded all 10 drafts, copies included.
- mistral left 1 of its 10 judgements unusable even after one repair.
- Prefix reuse is inferred from timing, not measured. Ollama's prompt-token
  counter reported the full 969 to 1,106 tokens on every classify call, yet
  those calls took 3.4 to 6.2 s, against 168 s for the 5,778-token guideline.

## 3. Results vs two baselines

Systems (`report/DECISIONS.md`, Evaluation scope): B0, a constant reply, and
B1, a TF-IDF nearest-neighbour copy of a past brand reply, both with zero LLM
calls; the no-RAG ablation; and the full system. Every score below comes from
the same 150 golden cases.

**Source for every number in this section**: `artifacts/eval/results.md` and
`results.json`, written by `python -m eval.run_eval` on 2026-09-19 with no
model call, from the traces of runs `20260919T051810Z` (B0, B1, no-RAG) and
`20260919T051814Z` (full) and 459 qwen2.5 judgements. Re-running it reproduces
the file.

> **Headline: the full system auto-handles 11.3% of tickets (17 of 150) inside
> the pre-registered safety budget** — zero SEVERE and zero ORDINARY harmful
> auto-replies, against a budget that allowed zero SEVERE and up to 7 ORDINARY.
> The no-RAG ablation reaches 14.0% (21 of 150) on the same budget. Both
> baselines are outside the budget entirely and have no comparable number.

**Comparing systems.** A difference is tested with a paired bootstrap: 2,000
resamples of the golden cases with a fixed seed, both systems scored on the
same resampled cases, and the 95% interval of the difference. A claimed
improvement stands only if that interval excludes zero. Overlapping per-system
intervals are not a test.

### 3.1 Intent classification

- **Ground truth: the 43 human labels only**, the 40 blind audit labels plus
  the 3 I labelled. The other 107 intent labels are phi3's own answers to the
  classifier's prompt, so scoring the phi3 systems against them would compare
  phi3 with itself (Section 5).
- **What:** macro-F1 over the 9 intents, per-class F1 with support, and the
  9 x 9 confusion matrix (in `artifacts/eval/results.md`).
- **Why macro-F1:** every intent counts equally, so a system cannot score well
  by getting only the frequent intents right.
- **Uncertainty:** a 95% bootstrap interval from 2,000 resamples of the 43
  human-labelled cases. At n = 43 these intervals are wide, and the report says
  so rather than letting a reader assume a ranking.
- **Reported twice:** stratified, on the 43 cases as sampled; and reweighted to
  the estimated population intent shares, each case weighted by its intent's
  population share over that intent's share of the scored cases. The population
  shares are phi3's estimate over the sampler's 1,500-case pool, so the
  reweighted figure inherits that estimate's error.

| intent classification | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| macro-F1, stratified [95% CI] | 0.014 [0.000, 0.031] | 0.350 [0.174, 0.425] | 0.526 [0.391, 0.658] | 0.526 [0.391, 0.658] |
| macro-F1, reweighted [95% CI] | 0.038 [0.000, 0.070] | 0.390 [0.211, 0.470] | 0.590 [0.434, 0.696] | 0.590 [0.434, 0.696] |

Per-class F1, with support among the 43 human labels:

| intent | support | B0 | B1 | no-RAG | full |
|---|---|---|---|---|---|
| billing_subscription | 7 | 0.000 | 0.667 | 0.727 | 0.727 |
| account_access | 1 | 0.000 | 1.000 | 0.500 | 0.500 |
| playback_failure | 4 | 0.000 | 0.000 | 0.444 | 0.444 |
| library_playlists | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| content_unavailable | 3 | 0.130 | 0.194 | 0.750 | 0.750 |
| feature_request | 8 | 0.000 | 0.500 | 0.824 | 0.824 |
| followup_diagnostic | 7 | 0.000 | 0.222 | 0.462 | 0.462 |
| chatter_thanks | 6 | 0.000 | 0.000 | 0.667 | 0.667 |
| other_unclear | 4 | 0.000 | 0.571 | 0.364 | 0.364 |

- **The full system and the ablation are identical here, to three decimals.**
  They share the classify stage; retrieval happens after the intent is chosen
  and cannot change it. The paired difference is +0.000 [0.000, 0.000].
- **phi3 beats TF-IDF, and the difference stands**: +0.176 [0.044, 0.372] over
  B1. B1 beats B0 by +0.336 [0.160, 0.410]. Both hold up under the paired test.
- **`library_playlists` scores 0.000 for every system**, on 3 human-labelled
  cases: nothing predicted it where a human chose it. It is the intent the
  guideline hedges most (playlists lost to an intruder are `account_access`,
  playlist feature requests are `feature_request`), and the classifier appears
  to have learned the exceptions instead of the class.

### 3.2 Escalation

150 cases, **48 labelled escalate**, 23 of them SEVERE if auto-handled.

- **Harmful auto-reply:** a case the golden label marks escalate that the
  system auto-handled. Its rate is over all 150 cases, the same denominator as
  the auto-handle rate. SEVERE when the true intent is `billing_subscription`
  or the label's `compromised` field is true; ORDINARY otherwise.
- **Pre-registered budget**, fixed before any system was scored: zero SEVERE,
  and at most 5% of all cases ORDINARY, which at n = 150 allows 7.
- **Operating curve:** auto-handle rate against harmful rate, sweeping the
  intent-confidence threshold (0.50 to 0.95) and the retrieval-similarity
  threshold (0.50 to 0.90) over a grid, re-scored offline from the traces. The
  operating point is the grid point with the highest auto-handle rate that
  stays inside the budget. Full grids are in `artifacts/eval/results.md`.

| escalation | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| operating point (confidence, similarity) | none | none | 0.50, — | 0.50, 0.50 |
| within the budget | no | no | **yes** | **yes** |
| auto-handle rate at that point (headline) | 1.000 [1.000, 1.000] | 0.853 [0.793, 0.907] | 0.140 [0.087, 0.200] | **0.113 [0.067, 0.173]** |
| harmful, SEVERE (budget: 0) | 23 (15.3%) | 6 (4.0%) | **0** | **0** |
| harmful, ORDINARY (budget: at most 7) | 25 (16.7%) | 24 (16.0%) | **0** | **0** |
| reweighted auto / SEVERE / ORDINARY | 1.000 / 0.177 / 0.147 | 0.842 / 0.047 / 0.142 | 0.102 / 0.000 / 0.000 | 0.090 / 0.000 / 0.000 |
| escalate precision | not defined | 0.818 | 0.372 | 0.361 |
| escalate recall | 0.000 | 0.375 | 1.000 | 1.000 |

- **Both baselines are outside the budget**, so neither has a headline number.
  B0 auto-handles everything and sends 23 SEVERE harmful replies; B1 sends 6.
  That is the cost of having no escalation policy, in cases.
- **Zero SEVERE is a bound, not a zero.** 23 of the 150 cases are
  SEVERE-relevant, so by the rule of three the 95% upper bound on the SEVERE
  rate is about 3/23 = **13.0%**, not 0.
- **Recall is 1.000 because the ladder escalates 133 of 150 cases.** Precision
  0.361 is the real shape of the result: the system is nowhere near its harm
  budget and is escalating far more than it needs to. The binding constraint is
  not safety but confidence.
- **The budget never bound.** Every grid point, at every threshold pair, had
  zero harmful auto-replies; the operating point is simply the lowest-threshold
  corner. A curve that never touches its ceiling means the thresholds are set
  far inside the safe region, and the auto-handle rate is what pays.

**Layer attribution, full system, at its operating point:**

| layer | fired on | alone caught (of the 48 labelled escalate) |
|---|---|---|
| L1 regex | 5 | 4 |
| L2 intent policy | 45 | 28 |
| L3 gates | 48 | 0 |
| L4 phi3 judgement | 75 | 0 |

- **The deterministic layers do the safety work.** L1 and L2 alone account for
  32 of the 48 escalate-labelled cases; L3 and L4 never uniquely caught one.
  The model-based layer is the least load-bearing part of the safety design.
- That is partly by construction and is disclosed as such: L1's
  account-compromise regex and L2's follow-up rule are the same rules that
  produced the `compromised` and `escalate` fields on the 147 model labels
  (Section 2, Section 5).
- L4 was not asked where L1 or L2 fired, so "alone" there means alone among the
  layers that were consulted.

### 3.3 Reply quality

Five binary checks, one qwen2.5 7B call per reply. Each cell is the rate of
"yes"; a good reply answers **no** to `contains_unsupported_specific` and
**yes** to the other four. B0's constant reply was judged once per golden
intent and the verdict reused.

| reply quality (rate) | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| contains_unsupported_specific (lower is better) | 0.000 | 0.302 | 0.040 | 0.060 |
| advances_resolution | 0.000 | 0.027 | 0.213 | 0.393 |
| addresses_stated_problem | 0.353 | 0.181 | 0.680 | 0.700 |
| tone_appropriate | 1.000 | 0.879 | 0.993 | 1.000 |
| would_send_unedited [95% CI] (headline) | 0.127 [0.073, 0.180] | 0.074 [0.034, 0.120] | 0.633 [0.560, 0.713] | **0.693 [0.620, 0.767]** |
| usable judgements | 150 of 150 | 149 of 150 | 150 of 150 | 150 of 150 |

- **Both LLM systems beat both baselines, decisively.** full − B1 on
  would_send_unedited is +0.620 [0.539, 0.699] and full − B0 is +0.567
  [0.473, 0.660]; the no-RAG ablation's margins are the same size. These stand.
- **Retrieval's gain does not stand.** full − no-RAG is **+0.060 [−0.033,
  +0.147]**, an interval that includes zero. On this evidence RAG did not
  produce a reply-quality improvement that survives a paired test at n = 150.
- **Where retrieval does show up is `advances_resolution`: 0.393 against
  0.213.** Precedents give the model concrete steps to copy. It does not
  convert into a defensible would_send_unedited gain.
- **B1 is the worst system on the checks that matter** (0.074 would_send,
  0.302 unsupported specifics) while copying real brand replies verbatim. A
  reply that was right for another customer is usually wrong for this one.

### 3.4 Judge validation

**Not run.** Three of the four planned comparisons need 60 replies hand-scored
by me, blind to the system, and that has not been done; the fourth, qwen
re-judging at temperature 0 through `complete(bypass_cache=True)`, is
implemented in the client but was not executed. Running mistral on a different
40 pairs than the pre-registered sample would not answer the question it was
pre-registered to answer, so it was left undone rather than substituted.

| comparison | checks | kappa | raw agreement |
|---|---|---|---|
| qwen vs my hand scores (60) | each of the 5 | not run | not run |
| mistral vs qwen (40) | each of the 5 | not run | not run |
| mistral vs my hand scores (40) | each of the 5 | not run | not run |
| qwen vs itself, temperature 0 (40) | all 5 | not applicable | not run |

**What is known about the judge, measured on this run:** 1 of 459 judgements
was unusable after its repair attempt (0.2%), and **130 of the 910 failing
answers (14%) arrived without the quote the rubric requires**. The rubric
demands evidence for every failing answer precisely so a human can check it;
one failing answer in seven cannot be checked. Every number in 3.3 rests on a
single 7B judge whose agreement with a human is unknown, and that is the
weakest link in this report (Section 5).

### 3.5 Cost and throughput

Median seconds per call, from `cache_hit=False` rows only: a cached call
measures a disk read, not inference. Tickets per hour counts the stages a live
system runs per ticket — classify, layer 4 where it is asked, retrieval and the
draft. Judging is evaluation only and is excluded.

| stage | model | calls | median s/call |
|---|---|---|---|
| classify | phi3 3.8B | 150 | 4.2 |
| escalation layer 4 | phi3 3.8B | 100 of 150 | 4.1 |
| draft, no-RAG | llama3 8B | 151 | 10.5 |
| draft, full | llama3 8B | 152 | 22.5 |
| judge, case-only evidence | qwen2.5 7B | 150 | 21.4 |
| judge, with precedents | qwen2.5 7B | 150 | 35.0 |

| cost and throughput | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| seconds per ticket | 0.000 | 0.005 | 18.7 | 31.5 |
| tickets per hour, this machine | effectively unbounded | effectively unbounded | 193 | 114 |
| cost of the same volume, hosted small model | not computed | not computed | not computed | not computed |

- **Retrieval doubles the draft cost**, 22.5 s against 10.5 s, because the
  prompt carries five precedents. Combined with 3.3, that is the ablation's
  verdict: RAG costs 69% more seconds per ticket and 2.7 auto-handled cases per
  100, for a quality gain that does not survive a paired test.
- **A total in `results.md` is not trustworthy, and the medians are.** The
  no-RAG judge stage shows 82,901 s of total latency because the machine slept
  mid-run and `perf_counter` counts suspended time. Per-call medians are
  unaffected; only sums that span a suspend are wrong.
- The hosted-model comparison was not computed: it needs a named model and a
  dated price, and inventing either would be a fabricated number.

### 3.6 Reference similarity (diagnostic only, never a headline)

ROUGE-L F1 against the brand's real reply, over 150 cases. The reference is
stripped of its leading handle and the agent's sign-off initials, neither of
which any system emits.

| reference similarity | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| ROUGE-L vs brand_reply [95% CI] | 0.146 [0.132, 0.159] | 0.251 [0.217, 0.291] | 0.147 [0.133, 0.163] | 0.233 [0.206, 0.260] |

**This table is why the metric is a diagnostic.** B1 scores highest, 0.251, by
copying a real brand reply verbatim — and B1 is the worst system in 3.3, at
0.074 would_send_unedited against the full system's 0.693. Ranking by
similarity to the reference would invert the ranking that matters. Finding 1.1
is the reason: 31.6% of the brand's visible replies move the customer to DM, so
the reference is often a deflection, and imitating it well is not answering
well. B0's constant reply is not a DM hand-off, and it still scores 0.146.

## 4. Failure analysis: top 5 failure modes

Five modes. Modes 1 to 3 were found while building the pipeline, and two of
them are fixed. Modes 4 and 5 are measured over the 150-case runs of
2026-09-16. Mode 1's rate over all 150 comes from the judge's
`contains_unsupported_specific` verdicts (Section 3.3), since a regex cannot
count invented product claims: 9 of 150 for the full system (6.0%) and 6 of 150
for the no-RAG ablation (4.0%).

| # | failure mode | cases | example | cause |
|---|---|---|---|---|
| 1 | Invented product instructions | 9 of 150 full, 6 of 150 no-RAG, by the judge's unsupported-specific check | disabling the "Offline Mode" feature | the draft's own guard checks numbers, money and timelines, not invented settings |
| 2 | Invented prior context (fixed) | 1 case, before and after | "as mentioned in our previous conversation", on a case with no earlier turns | the precedents were headed "PAST REPLIES" and read as this conversation |
| 3 | Retrieval leakage at cosine 1.0000 (fixed) | 1 golden case, 5 precedents; 10 excluded | a two-word thanks matched to the same thanks from five other customers | MinHash and MiniLM have different blind spots on emoji |
| 4 | Re-asks for what the customer already gave | 3 of 150 full, 4 of 150 no-RAG; none auto-handled | "try restarting your device and then reinstalling the app again", to a customer who had already reinstalled | nothing checks the draft against the facts already in the message |
| 5 | Sends the `URL` placeholder to the customer | 9 of 150 full, 1 of them auto-handled; 6 of 150 no-RAG | "You can vote for the idea here: URL" | cleaning replaced every link with the literal token `URL`, which then travels from precedent to draft |

### 4.1 Invented product instructions, auto-sent

The draft model invents settings that do not exist and states them as fact.
Both examples were auto-handled: no layer of the ladder fired, and layer 4
answered `none`.

- Case 2363279, whose customer asked how to stop Spotify updating on a Galaxy
  S7 Edge: *"Hey there! Sorry to hear that you're experiencing issues with
  Spotify updates on your Galaxy S7 Edge. To prevent updates from deleting your
  locally downloaded music, you can try disabling the "Offline Mode" feature."*
  Disabling Offline Mode does not prevent updates from deleting downloads.
- Case 790770, whose customer asked for a confirmation step before deleting
  downloads: *"Hey! We understand your concern. We'll pass this feedback along
  to our team. In the meantime, you can try enabling the "Confirm before
  deleting" option in your account settings. This might help prevent accidental
  song removal."* There is no such option; the customer had just written in to
  ask for one.

**Cause.** `src/generate.py` rejects a draft that states a number, an amount of
money, a date or a timeline absent from the precedents, and retries it once. An
invented setting name contains none of those, so it passes every check. The
precedents ground tone and format; they do not ground product truth, and
nothing in this project holds a list of Spotify's real settings to check
against.

**What would catch it.** Only the judge's `contains_unsupported_specific` check
(Section 3.3), which is why that check's own validation against hand scores
matters. Across the full run it flagged 9 of the full system's 150 replies
(6.0%) and 6 of the ablation's (4.0%) — and its own agreement with a human is
unmeasured (Section 3.4), so that rate is a signal, not a count.

### 4.2 Invented prior context, since fixed

Case 2363279 has no earlier turns at all. The first version of the draft
prompt produced: *"Hey there! Unfortunately, we don't have a specific setting to
prevent Spotify updates on your Galaxy S7 Edge. However, you can consider
blocking Spotify notifications at the system level, as mentioned in our
previous conversation."*

There was no previous conversation. The model had read a retrieved precedent,
another customer describing how they had blocked Spotify's notifications at the
system level, as something this customer had been told.

**Cause.** The precedent block was headed `=== PAST REPLIES ===` and listed
each precedent's customer message as well as the brand's reply, with nothing
saying whose conversation it was.

**Fix**, in `src/generate.py`: the section is now headed `=== OTHER CUSTOMERS'
PAST CASES (not this conversation) ===`, each line reads "Another customer
wrote" or "The brand replied", the case's own turns sit under `=== THIS
CONVERSATION: EARLIER TURNS ===`, and a rule in the system message forbids
referring to those cases as anything this customer said or was told. After the
change, the same case drafted a reply with no reference to a prior
conversation. The five smoke replies contained no other invented context.

**It replaced one invention with another.** The fixed reply is the "Offline
Mode" quote in 4.1. Grounding the model in whose conversation it is reading did
not stop it inventing the product.

### 4.3 Retrieval leakage at cosine 1.0000, caught by the test that exists for it

Golden case 507403 is a two-word thanks followed by two emoji. Its five nearest
precedents were the same two-word thanks from five other customers, with
different emoji, at cosine 1.0000 to four decimal places.

**Cause.** Two near-duplicate detectors with different blind spots. MinHash
over character 5-grams scored those pairs at Jaccard 0.47 to 0.70, below the
0.85 threshold, because in a string that short the emoji are much of the text.
MiniLM does not distinguish the emoji at all and embeds the messages
identically. A near-duplicate that one detector cannot see is invisible to the
pipeline that relies on it.

**Why it matters.** Those precedents carry the brand's real reply to what is,
in substance, the same message. Answering from them would produce a reply that
looks excellent and inflates every quality metric, while measuring nothing
except that the corpus repeats itself.

**Caught by** `tests/test_index.py`, on its first run, through the assertion
that no golden case may retrieve a precedent at 0.98 or above. **Fixed** by
adding embedding similarity as a third index exclusion, alongside `thread_id`
and `dup_group_id`: 10 precedents removed, leaving 39,086. No other golden case
came within 0.95 of a precedent; the next highest was 0.94.

### 4.4 Re-asks for what the customer already gave

The draft asks for a detail the customer has already supplied, or suggests a
step they have already reported trying. Counted by a keyword rule over the
150-case runs of 2026-09-16: the reply asks for a device, version or operating
system while the message already names one. **3 of 150 in the full system, 4 of
150 in the no-RAG ablation. None was auto-handled**, so in this run the mode
appears in drafts a human would have seen, not in replies sent to a customer.

The clearest instance is case 831462, whose customer reported that sharing
crashes the app on Android and said they had already reinstalled it, without
success. The no-RAG reply was:

> Sorry to hear that! Can you please try restarting your device and then
> reinstalling the app again? If the issue persists, please provide more
> details about your Android device and its OS version.

Reinstalling is the one thing the customer said they had done. The full system,
given precedents, asked the same customer to DM their Android OS and device
model instead: better, but still asking for a platform they had named.

**Cause.** The prompt carries the customer's message, and nothing checks the
finished draft against the facts already in it. The draft checks in
`src/generate.py` look for invented specifics, not for redundant ones.

**The count is soft, in both directions.** The keyword rule over-counts, since
asking for an app version the customer never gave is legitimate: of the 3 full
system hits, one (831462) is clear-cut and two are arguable. It also
under-counts, because a reply that repeats a step in different words does not
match. Read 3 of 150 as the order of magnitude, not a rate.

### 4.5 Sends the `URL` placeholder to the customer

`src/clean.py` replaces every link in the corpus with the literal token `URL`,
in the customer's message and in the brand's reply alike. Nothing turns it back
into a link, and nothing forbids it in a draft, so the token reaches the
customer. **9 of 150 in the full system, 1 of them auto-handled; 6 of 150 in
the no-RAG ablation.**

The auto-handled one, case 930008, asked for a feature to filter out live
versions of songs. The reply, sent with no escalation and layer 4 answering
`none`:

> We can see how that'd be useful! You can vote for the idea here: URL

It came from precedent 885112, whose real brand reply points the customer at
the official idea board, with the same placeholder standing where the link had
been. The agent did exactly what it was asked to do, grounding its reply in a
precedent, and the placeholder rode in with the wording. The customer is
directed to a link that does not exist.

The escalated instances are the mirror image: case 664584 said they had
forgotten to attach a screenshot, their own link already replaced by the token,
and was asked *"Can you please share the screenshot URL with us?"* — for
something they had just sent.

**Cause.** A preprocessing decision made for clustering and embedding leaked
into generation. Placeholders are the right call for similarity: a raw URL is
noise in an embedding. They are the wrong thing to hand a customer, and no
stage between cleaning and sending owns that distinction.

**Not fixed.** The cheap guard is to reject any draft containing `URL` or
`@USER`, the same way echoes are rejected, and escalate if the retry also
contains one; the real fix is to keep the original link beside the placeholder
at cleaning time so a reply can cite a real one. Both are Section 6 work.

### 4.6 A note on template collapse

**81 of the 150 no-RAG replies open with "Sorry to hear" (54%); with precedents
that falls to 24 of 150 (16%).** Retrieval is doing something beyond grounding:
it is supplying variety the model does not generate on its own. Not a failure
mode on its own, but a reply that opens identically every time reads as a bot
to the customer, and it is the ablation's most visible signature.

## 5. What is misleading about my headline number

The headline is **11.3%: the full system auto-handled 17 of 150 cases with zero
harmful auto-replies**. Four things make that number weaker than it looks, and
findings 1.1, 1.2 and 2.2 bear on all of them.

**The budget never bound, so the headline is not a safety result.** Every point
on the threshold grid had zero harmful auto-replies (Section 3.2). The system
is not trading harm for deflection at its operating point; it is escalating 133
of 150 cases, at 0.361 precision, and the auto-handle rate is what that
caution costs. A reader who sees "11.3% within budget" may assume the budget
was the constraint. It was not.

**The comparison the project was built to make came out null.** Retrieval's
effect on reply quality is +0.060 [-0.033, +0.147] against the ablation: the
interval includes zero, so on this evidence RAG did not improve reply quality.
It did cost 2.7 auto-handled cases per 100 and 69% more seconds per ticket.
The honest reading is that the ablation, not the full system, is the better
configuration on these 150 cases.

**A zero is a bound, not a zero.** Zero observed SEVERE harmful auto-replies
in N budget-relevant cases does not mean the true rate is zero: by the rule of
three, the 95% upper bound on that rate is about 3/N. N is the number of
golden cases labelled escalate whose intent is billing_subscription or whose
`compromised` field is true: **N = 23**, so the 95% upper bound on the SEVERE
rate is 3/23, about **13.0%**, not zero. Any zero reported here is stated in
the same sentence as N and that bound.

**The evaluation set is MODEL-LABELLED, and a third of the labels I checked
were wrong.** 147 of the 150 golden cases carry phi3's intent plus rule-derived
escalate and compromised fields; 3 I labelled myself. I then labelled 40 of the
147 independently and blind, seeing only the customer's message and its earlier
turns.

- **Agreement with the model labels: 25 of 40, 62.5%, Wilson 95% interval
  47.0% to 75.8%** (`data/golden/labeling_notes.md`, written by
  `python -m eval.label_stats`).
- So roughly one label in three is not what a careful reader would choose.
  Scoring macro-F1 against all 150 would measure agreement with phi3 rather
  than correctness, and for the phi3 systems it is worse: a model label is
  phi3's answer to the classifier's own prompt, so on those cases the
  prediction is the label itself. **Intent is therefore scored only on the 43
  human-labelled cases** (Section 3.1), with the loss of power n = 43 brings.
- **The disagreement is not spread evenly.** `other_unclear` agreed on 2 of 6
  (33.3%): phi3 reaches for it where I did not. `feature_request` agreed on 6
  of 7 (85.7%). Intents with fewer than 5 audited cases are not reported.
- The escalate labels on the 147 come from rules, and L1 and L2 of the ladder
  use those same rules (Section 2), so that agreement is construction. The
  audit covered intent only and does not repair it.

**The first audit pass was invalid, and the number it produced looked
plausible.** Its 40 answers stepped 1, 2, ... 9 straight through the keypad and
repeated, all 40 entered in 97 seconds. It reported 15.0% agreement, which
reads like a finding about a weak classifier rather than like an empty file:
near the 11% that guessing over 9 intents gives, which is what raised the
suspicion. A shift test settled it, joining each answer to the queue case
`shift` places away: shift 0 scored 15% and the four neighbours 8% to 11%, so
the join was sound and the answers themselves were the problem. A real
off-by-one would have shown a high-agreement neighbour.

`python -m src.label audit` now refuses to write once 9 answers in a row step
through the keypad, holds a file already on disk to a stricter run of 5, warns
whenever an answer arrives in under 2 seconds, and records how long each answer
took. The valid pass took 20.8 minutes, a median of 18.2 seconds per case.
**Annotation tooling needs adversarial checks against its own operator**: the
person best placed to corrupt a label set quietly is the one holding the
keyboard, and the corrupted number arrives looking like a result.

## 6. What I'd do next with one more week

In priority order. The first two buy measurement, not capability, which is
where this project's weakest claims are.

1. **Hand-label all 150 cases instead of auditing 40.** Intent is currently
   scored on 43 human labels because the other 107 are phi3's own answers to
   the classifier's prompt (Section 5). The blind audit found 62.5% agreement,
   so the model labels are not a usable ground truth, and n = 43 leaves the
   macro-F1 intervals wide enough that most system differences cannot be
   resolved. At the measured 18.2 seconds per case, 150 cases is about 45
   minutes of work, and it would move every intent number from "indicative" to
   "measured".

2. **Run the two judge-validation passes the harness is already built for**
   (Section 3.4): mistral 7B over the 40 (case, system) pairs drawn from the 60
   I hand-score, and qwen re-judging 40 replies through
   `complete(bypass_cache=True)` for self-consistency. Every reply-quality
   number rests on a single 7B judge whose agreement with a human is currently
   unknown. Kappa against my hand scores is what decides whether Section 3.3 is
   evidence or decoration.

3. **Replace the regex hallucination guard with an entailment check against
   the retrieved evidence.** Mode 4.1 is the worst failure in this report: the
   agent invented two Spotify settings and both were auto-sent. The guard in
   `src/generate.py` catches numbers, money, dates and timelines because those
   are what a regex can see. The right check asks a model whether each claim in
   the draft is entailed by the precedents, which is the same shape as the
   judge's `contains_unsupported_specific` but run before sending rather than
   after. It costs one extra call per draft and would have caught both.

4. **Re-induce the taxonomy for email rather than Twitter.** The nine intents
   come from 280-character public tweets, and two of them, `other_unclear` and
   `followup_diagnostic`, exist because the channel truncates context: a
   screenshot with no words, a reply whose meaning sits in the turns above it.
   A shared inbox gets subject lines, quoted history and attachments, so those
   two intents would split differently and the 29.8% structural ceiling
   (finding 1.2) would move. The pipeline transfers; the label set does not.

5. **Measure the calibration of the classifier's confidence.** Layer 3 gates on
   phi3's own probability at a 0.50 threshold, and the operating curve sweeps
   it, but nothing yet shows that a 0.6 means what a 0.9 means. A reliability
   curve over the human-labelled cases, plus the Brier score, would say whether
   the gate is a threshold on knowledge or on the model's habits. Finding 2.2
   is the warning: langdetect reported 1.0 confidence on plain English it had
   mislabelled, and a confident wrong answer from a 3.8B classifier is the same
   failure wearing a different hat.
