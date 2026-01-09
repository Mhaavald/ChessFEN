# Update version.json with build information
# This script should be run during deployment

param(
    [string]$Version = "1.0.0",
    [string]$BuildNumber = $null
)

# Get git commit hash (short)
$commit = "unknown"
try {
    $commit = git rev-parse --short HEAD 2>$null
    if (-not $commit) {
        $commit = "unknown"
    }
} catch {
    $commit = "unknown"
}

# Generate build number from timestamp if not provided
if (-not $BuildNumber) {
    $BuildNumber = Get-Date -Format "yyyyMMdd-HHmmss"
}

# Get current timestamp in ISO format
$timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"

# Create version object
$versionInfo = @{
    version = $Version
    build = $BuildNumber
    timestamp = $timestamp
    commit = $commit
}

# Write to version.json
$versionPath = Join-Path (Join-Path $PSScriptRoot "..") "version.json"
$versionInfo | ConvertTo-Json -Depth 10 | Set-Content $versionPath -Encoding UTF8

Write-Host "[OK] Updated version.json:"
Write-Host "  Version: $Version"
Write-Host "  Build: $BuildNumber"
Write-Host "  Commit: $commit"
Write-Host "  Timestamp: $timestamp"
