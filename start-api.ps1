# Start .NET API (HTTP + HTTPS for mobile)
cd $PSScriptRoot\web\ChessFEN.Api\ChessFEN.Api

$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback" -and $_.IPAddress -notmatch "^169" } | Select-Object -First 1).IPAddress

Write-Host "Starting API..." -ForegroundColor Cyan
Write-Host "  HTTP:  http://localhost:5001" -ForegroundColor Green
Write-Host "  HTTPS: https://localhost:5002" -ForegroundColor Green
if ($localIP) {
    Write-Host "  Mobile: https://${localIP}:5002" -ForegroundColor Magenta
}

dotnet run --urls="https://0.0.0.0:5002;http://0.0.0.0:5001"
