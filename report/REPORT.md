# Report: an evaluated local triage agent for SpotifyCares

Status: skeleton, 2026-09-13. Every number below was measured in this
repository and names its source. "TBD" marks every number that does not exist
yet. Nothing here is estimated or projected.

## 1. Problem framing

What "good" means for SpotifyCares: TBD.

### Measured constraints

**1.1 A third of the brand's own replies send the customer to DM.** 31.6% of
SpotifyCares' replies (13,124 of 41,497 cases) mention "DM" or "direct
message". That is a floor on replies that move the conversation out of the
public thread: the check only sees those two phrases, so deflection to phone,
chat or links is invisible to it. Any metric that compares a drafted reply
with the brand's real reply inherits this base rate. How the trivial baseline
scores against it is TBD (Section 3).

- Source: `python -m src.ingest` summary (commit cdd23c5); the matching rule is
  in `report/DECISIONS.md`, Ingest.

**1.2 Policy puts part of the traffic beyond any message-only agent.** In the
4,000-message clustering sample, the clusters merged into
billing_subscription hold 713 messages (17.8%). That intent always escalates:
resolving it needs account and payment data the agent has no access to. The
clusters merged into followup_diagnostic hold 478 (12.0%): replies whose
meaning lives in the earlier turns, which a message-only agent cannot act on.
Together that is 1,191 of 4,000 (29.8%). The ceiling on the auto-handle rate
is set by the traffic and the escalation policy before any modelling.
These are cluster shares, not labels: clusters are impure, the sample holds
one message per near-duplicate group, and the guideline moves cluster 1's
resolution confirmations to chatter_thanks. The labelled shares are TBD
(golden set).

- Source: `artifacts/clusters_summary.md` (commit 105a6b3); the handling
  policies are in `docs/taxonomy.md`.

### What I chose not to build

- No multi-turn dialogue policy.
- No fine-tuning.
- No tool execution.
- No live service.
- No sentiment feature for escalation, deliberately: anger and risk are
  different variables.

## 2. System

Architecture: TBD. The evaluation harness is not built yet.

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
assumed: TBD.

- Source: `python -m src.clean` output before and after the switch to
  `detect_langs` (commit 07a864f), compared on 2026-09-11.

**2.3 The guideline written for people is too large to be the prompt.**
`docs/taxonomy.md` is 6,259 phi3 tokens (5,778 before its Summary lines were
added), and one call evaluating it took 178 s on CPU. The compact block
rendered from the same file is 921 tokens. With a 2,048-token context, Ollama
read only 1,026 tokens of the 6,259-token guideline and returned no error.
Whether the compact block classifies as well as the full guideline is TBD.

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
150 tokens) has not been timed: TBD.

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
calls; the no-RAG ablation; and the full system. Every score comes from the
golden set: 150 cases (120 stratified by estimated intent, 30 hard cases).
147 are model-labelled by phi3 plus rules and 3 are labelled by me; 40 of
the 147 are audited blind by me (Section 5, `data/golden/labeling_notes.md`).

Every value in this section is TBD. The section fixes what each metric is,
why it was chosen and how it is reported before any system is scored, and the
safety budget in 3.2 is pre-registered.

**Comparing systems.** A difference between two systems is tested with a
paired bootstrap: 2,000 resamples of the golden cases with a fixed seed, the
same resampled cases scored for both systems, and the 95% interval of the
difference between them. A claimed improvement stands only if that interval
excludes zero. Overlapping per-system confidence intervals are not a test,
and at n = 150 that distinction decides whether any claimed improvement
stands.

### 3.1 Intent classification

- **Ground truth: the 43 human labels only**, my 40 blind audit labels plus
  the 3 cases I labelled. The other 147 intent labels are phi3's own answers
  to the classifier's prompt, so scoring the phi3 systems against them would
  compare phi3 with itself (Section 5).
- **What:** macro-F1 over the 9 intents, per-class F1 with its support (the
  number of human-labelled cases of that class), and the 9 x 9 confusion
  matrix.
- **Why macro-F1:** every intent counts equally, so a system cannot score
  well by getting only the frequent intents right.
- **Uncertainty:** a 95% bootstrap confidence interval on each system's
  macro-F1, from 2,000 resamples of the 43 human-labelled cases with a fixed
  seed. With n = 43, most differences between systems will be statistically
  indistinguishable, and the report says so rather than letting a reader
  assume a ranking. Differences are tested as above, never by comparing these
  intervals.
- **Reported twice:**
  - Stratified: on the 43 human-labelled cases as sampled.
  - Reweighted to the estimated population intent shares. Each case is
    weighted by the population share of its labelled intent divided by that
    intent's share of the golden set. The population shares are phi3's
    estimate over the sampler's 1,500-case pool (values TBD), so the
    reweighted figure inherits that estimate's error.
  - The golden set is deliberately not distribution-matched, so the
    stratified figure is not a production estimate.
- **Applies to** the systems that predict an intent: the no-RAG ablation and
  the full system, and B0 and B1 as well: B0 outputs the majority intent and
  B1 a TF-IDF classifier's intent (`report/DECISIONS.md`, Pipeline).

| intent classification | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| macro-F1, stratified [95% CI] | TBD | TBD | TBD | TBD |
| macro-F1, reweighted [95% CI] | TBD | TBD | TBD | TBD |

Per-class F1 with support, and the confusion matrix: TBD.

### 3.2 Escalation

Escalation is a threshold decision with asymmetric costs: auto-handling a case
that needed a human is worse than escalating one that did not. A single F1
treats the two errors alike, so it is the wrong summary.

- **Harmful auto-reply:** a case the golden label marks escalate that the
  system auto-handled.
  - Its rate is harmful auto-replies divided by all golden cases: a
    per-ticket rate on the same denominator as the auto-handle rate. Dividing
    by escalate cases only would restate 1 - recall.
  - SEVERE when the true intent is billing_subscription or the golden label's
    `compromised` field is true; ORDINARY otherwise. Both rates are reported.
  - On the 147 model-labelled cases, escalate and compromised come from
    rules applied to phi3's intent (Section 5), so a system whose escalation
    uses the same rules agrees with them by construction.
- **Precision and recall on the escalate class**, with escalate as the
  positive class.
- **Pre-registered safety budget**, fixed before any system is scored so the
  headline cannot be tuned after the fact:
  - SEVERE harmful auto-replies: zero tolerance. Layers 1 and 2 of the
    escalation stack are deterministic and exist to make this achievable by
    construction (layer definitions: Section 2, TBD).
  - ORDINARY harmful auto-replies: at most 5% of all golden cases. At n = 150
    one case is 0.67 percentage points, so the budget is coarse: 5% of 150 is
    7.5 cases, which allows at most 7, and the operating point is chosen at
    that one-case resolution.
  - The budget is checked on the golden set as labelled, in case counts. The
    reweighted rates are reported beside it.
- **Operating curve:** auto-handle rate on the x-axis against
  harmful-auto-reply rate on the y-axis, sweeping the intent-confidence and
  retrieval-similarity thresholds over a grid. Every grid point is plotted and
  the frontier drawn. Only a system that computes a threshold has a curve:
  the full system sweeps both, the no-RAG ablation has intent confidence
  only, and B0 and B1 are single points.
  - The operating point is the grid point with the highest auto-handle rate
    that stays within the budget. Chosen point: TBD.
  - Intent confidence is the probability phi3 gave the intent it wrote,
    from Ollama's token logprobs (`report/DECISIONS.md`, Pipeline).
    Finding 2.2 is why its calibration is checked on the golden set rather
    than assumed.
- **Headline metric: auto-handle rate at that operating point.** Hiver sells
  a shared-inbox helpdesk, and deflection at a stated safety budget is the
  number that maps to their product.
- **Layer attribution:** for every escalation, record which of the four
  escalation layers fired (layer definitions: Section 2, TBD). Report, per
  layer, how many escalations it fired on and how many it alone caught. If the
  deterministic rules catch most escalations, that is a finding about how
  little of the safety comes from the model.

| escalation | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| auto-handle rate at the operating point (headline) | TBD | TBD | TBD | TBD |
| harmful auto-reply rate, severe (budget: 0) | TBD | TBD | TBD | TBD |
| harmful auto-reply rate, ordinary (budget: at most 5%) | TBD | TBD | TBD | TBD |
| escalate precision | TBD | TBD | TBD | TBD |
| escalate recall | TBD | TBD | TBD | TBD |

### 3.3 Reply quality

Five binary checks instead of a 1-5 scale: a 7B-class judge tends to bunch
holistic ratings together, which makes agreement on them meaningless
(`report/DECISIONS.md`, Evaluation metrics). qwen2.5 7B judges every reply;
B0's constant reply is judged once per intent and the verdict reused
(`report/DECISIONS.md`, Evaluation scope).

| check | "yes" means | a good reply answers |
|---|---|---|
| contains_unsupported_specific | the reply states a number, date, duration or policy that is absent from the evidence; the judge must quote the span, or answer no | no |
| advances_resolution | the reply gives a concrete step, rather than only moving the conversation to DM | yes |
| addresses_stated_problem | the reply responds to the problem the customer actually raised | yes |
| tone_appropriate | the reply is polite and in the brand's voice | yes |
| would_send_unedited | the reply could be sent as written, with no edits | yes |

- **Evidence** is the case itself (the customer message and its prior turns)
  plus the retrieved precedents. The no-RAG ablation retrieves nothing, so its
  evidence is the case alone.
- **Reported** as a per-system rate for each check. would_send_unedited is
  the headline quality metric and carries a 95% bootstrap CI (2,000
  resamples).

| reply quality (rate) | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| contains_unsupported_specific | TBD | TBD | TBD | TBD |
| advances_resolution | TBD | TBD | TBD | TBD |
| addresses_stated_problem | TBD | TBD | TBD | TBD |
| tone_appropriate | TBD | TBD | TBD | TBD |
| would_send_unedited [95% CI] (headline) | TBD | TBD | TBD | TBD |

### 3.4 Judge validation

- **Against a human:** I hand-score 60 replies on the same five checks,
  blind to which system produced each. The harness shuffles the 60 and strips
  every system identifier before presenting them.
- **Self-consistency:** qwen re-judges 40 replies at temperature 0 through
  `complete(..., bypass_cache=True)`, and the exact-agreement rate is
  reported. Without the bypass, an identical prompt at temperature 0 is served
  from the disk cache and agreement would be 100% by construction. The bypass
  neither reads nor writes the cache, so the re-run cannot replace the
  verdicts the metrics use. Non-determinism here bounds every quality number
  downstream.
- **Cross-judge:** mistral 7B judges 40 (case, system) pairs drawn from the
  60 replies I hand-score, so its agreement with me is computable. That fixes
  the mistral sample at 40 pairs in total. Reported: agreement with qwen and
  with my hand scores.
- **Kappa and raw agreement, always both, for every check.** Kappa collapses
  when a check is nearly always "yes", as tone_appropriate almost certainly
  will be, even when the two raters agree on almost every case. Reporting
  only kappa there would understate agreement; reporting only raw agreement
  would overstate it.
- A low kappa is reported as a finding, not hidden: it would mean the quality
  numbers are directional only.

| comparison | checks | kappa | raw agreement |
|---|---|---|---|
| qwen vs my hand scores (60) | each of the 5 | TBD | TBD |
| mistral vs qwen (40) | each of the 5 | TBD | TBD |
| mistral vs my hand scores (40) | each of the 5 | TBD | TBD |
| qwen vs itself, temperature 0 (40) | all 5 | not applicable | TBD (exact agreement) |

### 3.5 Cost and throughput

- **Median seconds per call by stage**, computed from `cache_hit=False` rows
  of `artifacts/llm_calls.jsonl` only: a cached call measures a disk read,
  not inference.
- **Tickets per hour on this machine**, for the stages a live system would
  run per ticket. Judging is evaluation only and is excluded.
- **Cost of the same volume on a hosted small model**: a named model's
  published per-token price on a stated date, applied to the measured prompt
  and output token counts. No hosted model is called. Model, price and total:
  TBD.

| cost and throughput | value |
|---|---|
| median s/call, classify / draft / judge | TBD |
| tickets per hour, this machine | TBD |
| cost of the same volume, hosted small model | TBD |

### 3.6 Reference similarity (diagnostic only, never a headline)

- ROUGE-L between each system's reply and the brand's real reply, reported
  once.
- It is reported with this caveat: 31.6% of the brand's real replies move the
  customer to DM (finding 1.1), so a constant DM hand-off scores well on
  ROUGE-L by construction. Whether B0's constant reply is one: TBD. Imitating
  an often-unhelpful reference is not quality.

| reference similarity | B0 | B1 | no-RAG | full |
|---|---|---|---|---|
| ROUGE-L vs brand_reply (diagnostic) | TBD | TBD | TBD | TBD |

## 4. Failure analysis: top 5 failure modes

TBD. This needs the labelled golden set and the systems' outputs.

| # | failure mode | cases | example | cause |
|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD |

## 5. What is misleading about my headline number

Mandatory. TBD: there is no headline number yet. Findings 1.1, 1.2 and 2.2
bear on it.

**A zero is a bound, not a zero.** Zero observed SEVERE harmful auto-replies
in N budget-relevant cases does not mean the true rate is zero: by the rule of
three, the 95% upper bound on that rate is about 3/N. N is the number of
golden cases labelled escalate whose intent is billing_subscription or whose
`compromised` field is true (value TBD). Any zero reported here is stated in
the same sentence as N and that bound.

**The evaluation set is MODEL-LABELLED, not hand-labelled.** 147 of the 150
golden cases carry phi3's intent plus rule-derived escalate and compromised
fields; 3 were labelled by me, one of them with a phi3 suggestion shown. 40
of the 147 were independently labelled by me, blind, and agreement with the
model labels is TBD (Wilson 95% interval TBD) until that audit is done. The
classifier and the labels come from the same model family and the same
taxonomy, so any agreement between them is partly shared error rather than
accuracy, and macro-F1 against these labels measures consistency with phi3
rather than correctness. For the phi3 systems it is stronger than that: a
model label is phi3's answer to the classifier's own prompt, so on those
cases the classifier's prediction is the label itself. Intent metrics are
therefore scored only on the 43 human-labelled cases (Section 3.1). The
escalate labels on the 147 come from rules, so a system whose escalation uses
the same rules agrees with them by construction; the audit checks the intent
only. `data/golden/labeling_notes.md` holds the audit figures, written by
`python -m eval.label_stats`.

## 6. What I'd do next with one more week

TBD.
