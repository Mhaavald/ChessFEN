# Start Python Inference Service
cd $PSScriptRoot
if (Test-Path ".venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }
$env:FLASK_DEBUG = "true"
python src\inference\inference_service.py
