# The full evaluation run (eval/run_all.py): B0 and B1, the no-RAG ablation, the full system, the qwen2.5 judge,
# then the metrics. Prints the projected wall clock, starts the run in the background and returns.
# Follow it with:  Get-Content artifacts\run.log -Wait -Tail 20
# Every LLM call is cached on disk, so running this again after a stop resumes where the run stopped.
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
. .\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = "1"  # MiniLM loads from the local Hugging Face cache; no run contacts the Hub
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"  # progress lines reach the log as they happen

python -m eval.run_all --projection-only
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }

$log = Join-Path $root "artifacts\run.log"
$errors = Join-Path $root "artifacts\run.err"
$python = (Get-Command python).Source  # the virtual environment's python, activated above
$process = Start-Process -FilePath $python -ArgumentList "-m", "eval.run_all" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $errors -PassThru
"Started the run: process $($process.Id). Progress: Get-Content artifacts\run.log -Wait -Tail 20"
Pop-Location
