# Start all servers as background processes
# These survive screen lock and run until manually stopped

$root = $PSScriptRoot

Write-Host "Starting Chess FEN servers..." -ForegroundColor Cyan

# Activate venv for Python processes
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
}

# Start Python inference
$inferenceProc = Start-Process -FilePath "python" -ArgumentList "src\inference\inference_service.py" -WorkingDirectory $root -PassThru -WindowStyle Minimized
Write-Host "  Inference (5000): PID $($inferenceProc.Id)" -ForegroundColor Green

Start-Sleep -Seconds 2

# Start .NET API  
$apiProc = Start-Process -FilePath "dotnet" -ArgumentList "run", "--urls=http://0.0.0.0:5001" -WorkingDirectory (Join-Path $root "web\ChessFEN.Api\ChessFEN.Api") -PassThru -WindowStyle Minimized
Write-Host "  API (5001): PID $($apiProc.Id)" -ForegroundColor Green

Start-Sleep -Seconds 2

# Start Blazor Debug UI (HTTP + HTTPS for mobile)
$blazorProc = Start-Process -FilePath "dotnet" -ArgumentList "run", "--urls=https://0.0.0.0:5004;http://0.0.0.0:5003" -WorkingDirectory (Join-Path $root "web\ChessFEN.Web\ChessFEN.Web") -PassThru -WindowStyle Minimized
Write-Host "  Blazor Debug (5003/5004): PID $($blazorProc.Id)" -ForegroundColor Green

Start-Sleep -Seconds 2

# Start HTTP client (localhost only)
$clientHttpProc = Start-Process -FilePath "python" -ArgumentList "web\ChessFEN.Client\serve.py", "5005" -WorkingDirectory $root -PassThru -WindowStyle Minimized
Write-Host "  Client HTTP (5005): PID $($clientHttpProc.Id)" -ForegroundColor Green

# Start HTTPS proxy client (for mobile)
$clientHttpsProc = Start-Process -FilePath "python" -ArgumentList "web\ChessFEN.Client\serve_proxy.py", "5006" -WorkingDirectory $root -PassThru -WindowStyle Minimized
Write-Host "  Client HTTPS (5006): PID $($clientHttpsProc.Id)" -ForegroundColor Green

# Get local IP
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback" -and $_.IPAddress -notmatch "^169" } | Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "=== All servers started (minimized windows) ===" -ForegroundColor Green
Write-Host ""
Write-Host "  Localhost URLs:" -ForegroundColor Cyan
Write-Host "    Client UI:    http://localhost:5005" -ForegroundColor White
Write-Host "    Blazor Debug: http://localhost:5003" -ForegroundColor White
Write-Host ""
Write-Host "  Mobile URLs (accept cert warnings):" -ForegroundColor Magenta
Write-Host "    Client UI:    https://${localIP}:5006" -ForegroundColor Green
Write-Host "    Blazor Debug: https://${localIP}:5004" -ForegroundColor Green
Write-Host ""
Write-Host "To stop all: .\stop-background.ps1" -ForegroundColor Yellow

# Save PIDs to file for later cleanup
@{
    Inference = $inferenceProc.Id
    Api = $apiProc.Id
    Blazor = $blazorProc.Id
    ClientHttp = $clientHttpProc.Id
    ClientHttps = $clientHttpsProc.Id
} | ConvertTo-Json | Set-Content (Join-Path $root "server-pids.json")

Write-Host ""
Write-Host "PIDs saved to server-pids.json" -ForegroundColor Gray
