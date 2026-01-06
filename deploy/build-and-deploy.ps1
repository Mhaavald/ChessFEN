# Build and deploy script for Azure Container Apps
# Updates version before building

param(
    [string]$Version = "1.0.0",
    [string]$ResourceGroup = "chess-fen-rg",
    [string]$ContainerApp = "chess-fen-app",
    [string]$Registry = "chessfen",
    [string]$ImageName = "chessfen-app"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ChessFEN Build & Deploy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Update version
Write-Host "[1/4] Updating version.json..." -ForegroundColor Yellow
$buildNumber = Get-Date -Format "yyyyMMdd-HHmmss"
& "$PSScriptRoot\..\scripts\update-version.ps1" -Version $Version -BuildNumber $buildNumber

# Step 2: Build Docker image
Write-Host ""
Write-Host "[2/4] Building Docker image..." -ForegroundColor Yellow
$imageTag = "$Registry.azurecr.io/$($ImageName):$buildNumber"
$imageLatest = "$Registry.azurecr.io/$($ImageName):latest"

Set-Location "$PSScriptRoot\.."
docker build -t $imageTag -t $imageLatest -f deploy/Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Image built: $imageTag" -ForegroundColor Green

# Step 3: Push to Azure Container Registry
Write-Host ""
Write-Host "[3/4] Pushing to Azure Container Registry..." -ForegroundColor Yellow
docker push $imageTag
docker push $imageLatest

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Docker push failed" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Image pushed successfully" -ForegroundColor Green

# Step 4: Update Azure Container App
Write-Host ""
Write-Host "[4/4] Updating Azure Container App..." -ForegroundColor Yellow
az containerapp update `
    --name $ContainerApp `
    --resource-group $ResourceGroup `
    --image $imageTag

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Container App update failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Version: $Version" -ForegroundColor Cyan
Write-Host "Build: $buildNumber" -ForegroundColor Cyan
Write-Host "Image: $imageTag" -ForegroundColor Cyan
Write-Host ""
