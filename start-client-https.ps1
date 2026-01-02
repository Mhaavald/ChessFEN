# Start Client UI (HTTPS with API Proxy - for mobile camera)
cd $PSScriptRoot
if (Test-Path ".venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }

$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback" -and $_.IPAddress -notmatch "^169" } | Select-Object -First 1).IPAddress

Write-Host "Starting HTTPS server with API proxy..." -ForegroundColor Cyan
Write-Host "Access from iPhone: https://${localIP}:5006" -ForegroundColor Green
Write-Host "(Accept certificate warning - only need to do it once!)" -ForegroundColor Yellow
Write-Host ""
Write-Host "This works like Blazor - API calls proxied server-side" -ForegroundColor Gray

python web\ChessFEN.Client\serve_proxy.py 5006
