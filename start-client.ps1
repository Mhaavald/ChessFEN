# Start Client UI (HTTP for localhost)
cd $PSScriptRoot
if (Test-Path ".venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
python web\ChessFEN.Client\serve.py 5005
