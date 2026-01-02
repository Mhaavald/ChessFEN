# Stop all background servers

$root = $PSScriptRoot
$pidFile = Join-Path $root "server-pids.json"

if (Test-Path $pidFile) {
    $pids = Get-Content $pidFile | ConvertFrom-Json
    
    Write-Host "Stopping servers..." -ForegroundColor Yellow
    
    foreach ($prop in $pids.PSObject.Properties) {
        try {
            Stop-Process -Id $prop.Value -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped $($prop.Name) (PID $($prop.Value))" -ForegroundColor Green
        } catch {
            Write-Host "  $($prop.Name) already stopped" -ForegroundColor Gray
        }
    }
    
    Remove-Item $pidFile
    Write-Host "Done" -ForegroundColor Green
} else {
    Write-Host "No server-pids.json found. Servers may not be running." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To manually find and stop:" -ForegroundColor Gray
    Write-Host '  Get-Process python, dotnet | Stop-Process' -ForegroundColor Gray
}
