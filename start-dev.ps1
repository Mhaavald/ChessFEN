# Chess FEN Scanner - Startup Script
# Run this from the repository root

Write-Host "=== Chess FEN Scanner ===" -ForegroundColor Cyan

# Check Python environment
Write-Host "`nChecking Python environment..." -ForegroundColor Yellow
if (Test-Path ".venv\Scripts\activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    Write-Host "  Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "  Warning: No .venv found, using system Python" -ForegroundColor Yellow
}

# Install Flask if needed
pip show flask 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nInstalling Flask dependencies..." -ForegroundColor Yellow
    pip install flask flask-cors
}

# Start Python inference service in background
Write-Host "`nStarting Python inference service on port 5000..." -ForegroundColor Yellow
$env:FLASK_DEBUG = "true"
$env:FLASK_PORT = "5000"
Start-Process -FilePath "python" -ArgumentList "src\inference\inference_service.py" -NoNewWindow

# Wait for service to start
Start-Sleep -Seconds 3

# Start .NET API
Write-Host "`nStarting .NET API on port 5001..." -ForegroundColor Yellow
Push-Location "web\ChessFEN.Api\ChessFEN.Api"
Start-Process -FilePath "dotnet" -ArgumentList "run", "--urls=http://localhost:5001" -NoNewWindow
Pop-Location

# Wait for API to start
Start-Sleep -Seconds 3

# Start Blazor Web
Write-Host "`nStarting Blazor Web frontend on port 5002..." -ForegroundColor Yellow
Push-Location "web\ChessFEN.Web\ChessFEN.Web"
Start-Process -FilePath "dotnet" -ArgumentList "run", "--urls=http://localhost:5002" -NoNewWindow
Pop-Location

Write-Host "`n=== All services started ===" -ForegroundColor Green
Write-Host "  Python Inference: http://localhost:5000" -ForegroundColor Cyan
Write-Host "  .NET API:         http://localhost:5001/swagger" -ForegroundColor Cyan
Write-Host "  Web Frontend:     http://localhost:5002" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop all services" -ForegroundColor Yellow

# Keep script running
while ($true) { Start-Sleep -Seconds 60 }
