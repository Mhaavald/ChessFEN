# Chess FEN Scanner - Client Startup Script
# Starts only the required services for the end-user client

Write-Host "=== Chess FEN Scanner - Client Mode ===" -ForegroundColor Cyan

# Check Python environment
Write-Host "`nChecking Python environment..." -ForegroundColor Yellow
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

if (Test-Path ".venv\Scripts\activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
    Write-Host "  Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "  Warning: No .venv found, using system Python" -ForegroundColor Yellow
}

# Start Python inference service
Write-Host "`nStarting Python inference service on port 5000..." -ForegroundColor Yellow
$env:FLASK_DEBUG = "false"
$env:FLASK_PORT = "5000"
Start-Process -FilePath "python" -ArgumentList "src\inference\inference_service.py" -NoNewWindow

Start-Sleep -Seconds 3

# Start .NET API
Write-Host "`nStarting .NET API on port 5001..." -ForegroundColor Yellow
Push-Location "web\ChessFEN.Api\ChessFEN.Api"
Start-Process -FilePath "dotnet" -ArgumentList "run", "--urls=http://0.0.0.0:5001" -NoNewWindow
Pop-Location

Start-Sleep -Seconds 3

# Start Client (static file server)
Write-Host "`nStarting Client on port 5004..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "web\ChessFEN.Client\serve.py", "5004" -NoNewWindow

Pop-Location

# Get local IP for mobile access
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback" -and $_.IPAddress -notmatch "^169" } | Select-Object -First 1).IPAddress

Write-Host "`n=== Services Started ===" -ForegroundColor Green
Write-Host "  Python Inference: http://localhost:5000" -ForegroundColor Cyan
Write-Host "  .NET API:         http://localhost:5001/swagger" -ForegroundColor Cyan
Write-Host "  Client App:       http://localhost:5004" -ForegroundColor Green
if ($localIP) {
    Write-Host "`n  Mobile access:    http://${localIP}:5004" -ForegroundColor Magenta
    Write-Host "  (Note: Camera requires HTTPS on mobile browsers)" -ForegroundColor Yellow
}
Write-Host "`nPress Ctrl+C to stop all services" -ForegroundColor Yellow

# Keep script running
while ($true) { Start-Sleep -Seconds 60 }
