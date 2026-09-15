# support-triage-agent

[![CI](https://github.com/PPaul14/support-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/PPaul14/support-triage-agent/actions/workflows/ci.yml)

An evaluated, fully local AI agent for first-line customer support triage.
It classifies intent, drafts a reply grounded in the brand history, and
decides auto-handle vs escalate with a stated reason. The evaluation is the
point of the project, not the agent.

Status: work in progress. No results yet.

## Setup (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull phi3:3.8b-mini-128k-instruct-q4_0
ollama pull llama3:8b-instruct-q4_0
ollama pull mistral:7b-instruct-v0.3-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M
```

All inference runs locally through Ollama. No hosted LLM APIs are used.

## Models

Four independent model families from four different labs are used, so the
draft model and the two judges never share weights: no judge grades its own
model's output.

| Role | Ollama tag | Family (lab) | Params | Quant | Manifest digest |
|---|---|---|---|---|---|
| classify + escalate | `phi3:3.8b-mini-128k-instruct-q4_0` | Phi-3 Mini (Microsoft) | 3.8B | Q4_0 | `sha256:4f222292793889a9a40a020799cfd28d53f3e01af25d48e06c5e708610fc47e9` |
| draft replies | `llama3:8b-instruct-q4_0` | Llama 3 (Meta) | 8.0B | Q4_0 | `sha256:365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1` |
| judge A | `mistral:7b-instruct-v0.3-q4_K_M` | Mistral 7B v0.3 (Mistral AI) | 7.2B | Q4_K_M | `sha256:6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf` |
| judge B | `qwen2.5:7b-instruct-q4_K_M` | Qwen2.5 (Alibaba) | 7.6B | Q4_K_M | `sha256:845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` |

A tag on the Ollama registry can be re-pointed later; the digest is the real
pin. After pulling, the `ID` column of `ollama list` must equal the first 12
characters of each digest. If it does not, you have different weights and the
results will not reproduce.

Separate weights do not guarantee independent errors: all four families were
pretrained on overlapping web data, so a judge can still share blind spots
with the draft model.

## Layout

- `src/` - pipeline stages; `src/llm.py` is the only module that calls Ollama
- `eval/` - evaluation and cost reporting
- `tests/` - unit tests
- `data/raw/` - source data (not committed)
- `data/sample/` - small sample for quick runs
- `data/golden/` - golden set: phi3 labels plus a 40-case blind human audit
  (`data/golden/labeling_notes.md`)
- `artifacts/` - LLM cache, call log, run traces
- `report/` - evaluation write-up

## License

The code is MIT licensed - see [LICENSE](LICENSE).

Quoted dataset excerpts are not MIT. The example customer messages in
`docs/taxonomy.md` and `artifacts/classifier_prompt.txt` come from the
Customer Support on Twitter dataset,
[thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter),
and are licensed CC BY-NC-SA 4.0. See [data/DATA_LICENSE.md](data/DATA_LICENSE.md).
