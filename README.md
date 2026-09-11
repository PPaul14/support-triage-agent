# support-triage-agent

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

ollama pull qwen2.5:3b-instruct
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct
ollama pull mistral:7b-instruct
```

All inference runs locally through Ollama. No hosted LLM APIs are used.

## Layout

- `src/` - pipeline stages; `src/llm.py` is the only module that calls Ollama
- `eval/` - evaluation and cost reporting
- `tests/` - unit tests
- `data/raw/` - source data (not committed)
- `data/sample/` - small sample for quick runs
- `data/golden/` - hand-labelled golden set
- `artifacts/` - LLM cache, call log, run traces
- `report/` - evaluation write-up

## License

MIT - see [LICENSE](LICENSE).
