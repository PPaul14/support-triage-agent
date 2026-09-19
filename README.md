# support-triage-agent

[![CI](https://github.com/PPaul14/support-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/PPaul14/support-triage-agent/actions/workflows/ci.yml)

A fully local triage agent for first-line customer support on public Twitter
threads: it classifies intent, drafts a reply grounded in the brand's own
history, and decides auto-handle or escalate with a stated reason — evaluated
against two baselines and an ablation on 150 golden cases, with the evidence in
[report/REPORT.md](report/REPORT.md).

## Results

150 golden cases, four systems, one local machine. The headline is the
auto-handle rate at a **pre-registered safety budget** — zero SEVERE harmful
auto-replies and at most 7 ORDINARY, fixed before any system was scored.

| | B0 constant | B1 TF-IDF copy | no-RAG | **full** |
|---|---|---|---|---|
| **auto-handle rate at the budget** | outside budget | outside budget | 14.0% | **11.3%** |
| harmful auto-replies: SEVERE / ORDINARY | 23 / 25 | 6 / 24 | **0 / 0** | **0 / 0** |
| would_send_unedited (qwen2.5 judge) | 0.127 | 0.074 | 0.633 | **0.693** |
| macro-F1, 43 human-labelled cases | 0.014 | 0.350 | 0.526 | 0.526 |
| seconds of local inference per ticket | ~0 | ~0 | 18.7 | 31.5 |
| ROUGE-L vs the brand's real reply (diagnostic) | 0.146 | **0.251** | 0.147 | 0.233 |

Three things this table is for:

- **The baselines are not close, and not safe.** B0 answers everything and
  sends 23 SEVERE harmful replies; B1 sends 6. Neither is inside the budget, so
  neither has a headline number at all.
- **Retrieval did not pay for itself.** full − no-RAG on reply quality is
  +0.060 [−0.033, +0.147] — a paired bootstrap interval that includes zero —
  while costing 2.7 auto-handled cases per 100 and 69% more inference per
  ticket. The ablation is the better configuration on this evidence.
- **The best ROUGE-L belongs to the worst system.** B1 copies real brand
  replies verbatim, tops the similarity column and comes last on
  would_send_unedited. That is why similarity is a diagnostic here and never a
  headline.

Full numbers, intervals and method: [report/REPORT.md](report/REPORT.md)
Section 3. Every figure is reproduced by `python -m eval.run_eval` from the
committed run traces, with no model calls.

## Quickstart (Windows, PowerShell)

```powershell
.\scripts\smoke.ps1      # 5 golden cases through all four systems, about 4 minutes
.\scripts\run_all.ps1    # the full 150-case run in the background, then the metrics
```

`run_all.ps1` prints its projected wall clock before starting and logs to
`artifacts\run.log` (`Get-Content artifacts\run.log -Wait -Tail 20`). Every LLM
call is cached on disk, so a stopped run resumes where it stopped. Both scripts
set `HF_HUB_OFFLINE=1`: MiniLM loads from the local Hugging Face cache and no
run touches the network.

<details>
<summary>First-time setup</summary>

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull phi3:3.8b-mini-128k-instruct-q4_0
ollama pull llama3:8b-instruct-q4_0
ollama pull mistral:7b-instruct-v0.3-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M

python -m src.ingest     # needs data/raw/twcs.csv, see What runs without the dataset
python -m src.clean
python -m src.index      # downloads MiniLM once, so this first run needs the network
```

</details>

## What runs without the dataset

The corpus is not redistributable, so `data/raw/twcs.csv` has to come from
[Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter),
which needs an account. Without it you can still:

- run the test suite — `python -m pytest -v -rs`; the tests that need the
  corpus skip and say so, which is exactly what CI does on every push;
- read [report/REPORT.md](report/REPORT.md),
  [report/DECISIONS.md](report/DECISIONS.md),
  [report/CREDITS.md](report/CREDITS.md) and
  [docs/taxonomy.md](docs/taxonomy.md), each number citing its source;
- inspect the golden set: `data/golden/` holds the case ids, the labels and the
  audit, with no message text in any of them.

What you cannot do without it: rebuild the corpus, the precedent index or any
metric. **One further gap, stated plainly:** the index and B1 train on phi3's
cached intent estimates for 1,500 cases, which live in `artifacts/llm_cache/`
and are not committed. That cache is rebuilt by running the pipeline, but there
is no single command that regenerates just those estimates, so a clean-room
reproduction of the index needs a phi3 pass this repository does not ship.

## Models

Four independent model families from four different labs, so the draft model
and the two judges never share weights: no judge grades its own model's output.

| Role | Ollama tag | Family (lab) | Params | Quant | Manifest digest |
|---|---|---|---|---|---|
| classify + escalate | `phi3:3.8b-mini-128k-instruct-q4_0` | Phi-3 Mini (Microsoft) | 3.8B | Q4_0 | `sha256:4f222292793889a9a40a020799cfd28d53f3e01af25d48e06c5e708610fc47e9` |
| draft replies | `llama3:8b-instruct-q4_0` | Llama 3 (Meta) | 8.0B | Q4_0 | `sha256:365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1` |
| judge A | `mistral:7b-instruct-v0.3-q4_K_M` | Mistral 7B v0.3 (Mistral AI) | 7.2B | Q4_K_M | `sha256:6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf` |
| judge B | `qwen2.5:7b-instruct-q4_K_M` | Qwen2.5 (Alibaba) | 7.6B | Q4_K_M | `sha256:845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` |

Embeddings are `sentence-transformers/all-MiniLM-L6-v2`, pinned to commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, on CPU.

A tag on the Ollama registry can be re-pointed later; the digest is the real
pin. After pulling, the `ID` column of `ollama list` must equal the first 12
characters of each digest. If it does not, you have different weights and the
results will not reproduce.

Separate weights do not guarantee independent errors: all four families were
pretrained on overlapping web data, so a judge can still share blind spots with
the draft model.

## Layout

- `src/` — pipeline stages; `src/llm.py` is the only module that calls Ollama
- `eval/` — metrics, the reply-quality judge and the run orchestrator
- `scripts/` — `smoke.ps1` and `run_all.ps1`
- `tests/` — unit tests; the corpus-dependent ones skip without the data
- `data/golden/` — the golden set: labels, audit and notes, no message text
- `docs/` — the labelling guideline ([taxonomy.md](docs/taxonomy.md)) and its
  [quick-reference](docs/labelling_cheatsheet.md)
- `artifacts/` — LLM cache, call log, traces, judgements, results (gitignored)
- `report/` — [the write-up](report/REPORT.md), the
  [decision log](report/DECISIONS.md) and [credits](report/CREDITS.md)

## Licence

**The code is MIT licensed** — see [LICENSE](LICENSE).

**The dataset excerpts are not.** The example customer messages quoted in
`docs/taxonomy.md` and `artifacts/classifier_prompt.txt` come from the Customer
Support on Twitter dataset
([thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter))
and are licensed CC BY-NC-SA 4.0 — non-commercial, share-alike. They are the
only message text in this repository: 46 quoted cases, used as the classifier's
few-shot examples. Traces, judgements and the LLM cache all contain message text
and are gitignored for that reason. See
[data/DATA_LICENSE.md](data/DATA_LICENSE.md).
