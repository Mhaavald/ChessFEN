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

# Start .NET API with HTTPS for mobile camera support
Write-Host "`nStarting .NET API on port 5001 (HTTPS: 5002)..." -ForegroundColor Yellow
Push-Location "web\ChessFEN.Api\ChessFEN.Api"
Start-Process -FilePath "dotnet" -ArgumentList "run", "--urls=https://0.0.0.0:5002;http://0.0.0.0:5001" -NoNewWindow
Pop-Location

# Wait for API to start
Start-Sleep -Seconds 3

# Start Blazor Web with HTTPS for camera support on mobile
Write-Host "`nStarting Blazor Web frontend on port 5003 (HTTPS: 5004)..." -ForegroundColor Yellow
Push-Location "web\ChessFEN.Web\ChessFEN.Web"
# Use 0.0.0.0 to allow connections from other devices on the network
Start-Process -FilePath "dotnet" -ArgumentList "run", "--urls=https://0.0.0.0:5004;http://0.0.0.0:5003" -NoNewWindow
Pop-Location

# Wait for Blazor to start
Start-Sleep -Seconds 3

# Start static client (lightweight end-user UI)
Write-Host "`nStarting Client UI on port 5005 (HTTPS: 5006)..." -ForegroundColor Yellow
Start-Process -FilePath "python" -ArgumentList "web\ChessFEN.Client\serve.py", "5005" -NoNewWindow
Start-Process -FilePath "python" -ArgumentList "web\ChessFEN.Client\serve_https.py", "5006" -NoNewWindow

# Get local IP for mobile access
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback" -and $_.IPAddress -notmatch "^169" } | Select-Object -First 1).IPAddress

Write-Host "`n=== All services started ===" -ForegroundColor Green
Write-Host "  Python Inference: http://localhost:5000" -ForegroundColor Cyan
Write-Host "  .NET API:         http://localhost:5001 | https://localhost:5002" -ForegroundColor Cyan
Write-Host "  Blazor Frontend:  http://localhost:5003 | https://localhost:5004" -ForegroundColor Cyan
Write-Host "  Client UI:        http://localhost:5005 | https://localhost:5006" -ForegroundColor Green
if ($localIP) {
    Write-Host "`n  Mobile access:" -ForegroundColor Magenta
    Write-Host "    Client UI: https://${localIP}:5006" -ForegroundColor Green
    Write-Host "    Blazor:    https://${localIP}:5004" -ForegroundColor Cyan
    Write-Host "  (Accept certificate warning on phone)" -ForegroundColor Yellow
}
Write-Host "`nFor mobile camera access, use HTTPS URLs" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow

# Keep script running
while ($true) { Start-Sleep -Seconds 60 }
