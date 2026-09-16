# Smoke test: the first 5 golden cases through B0 and B1, the no-RAG ablation and the full system.
# Usage, from the repository root:  .\scripts\smoke.ps1
Push-Location (Split-Path $PSScriptRoot -Parent)
. .\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = "1"  # MiniLM loads from the local Hugging Face cache; no run contacts the Hub
$env:PYTHONIOENCODING = "utf-8"

python -m src.baselines --limit 5
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
python -m src.pipeline --limit 5 --no-rag
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
python -m src.pipeline --limit 5
$status = $LASTEXITCODE
Pop-Location
exit $status
