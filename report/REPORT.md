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

Systems compared (`report/DECISIONS.md`, Evaluation scope): B0, a constant
reply, and B1, a TF-IDF nearest-neighbour copy of a past brand reply, both
with zero LLM calls; the no-RAG ablation; and the full system. Scores are to
be reported both stratified and reweighted to the traffic mix, because the
golden set is deliberately stratified rather than distribution-matched.

- Golden set: 150 cases planned (120 stratified by estimated intent, 30 hard
  cases). Labelled: TBD.
- Metrics and scores: TBD.

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

## 6. What I'd do next with one more week

TBD.
