<#
Pipeline: (1) Grayscale (2) Contrast stretch (3) Sharpen (unsharp)
Outputs: PNG files to an output folder (keeps originals unchanged)

Usage examples:
  .\prep-chess.ps1 -InputPath "C:\imgs" -OutputPath "C:\out"
  .\prep-chess.ps1 -InputPath "C:\imgs" -OutputPath "C:\out" -Recurse
  .\prep-chess.ps1 -InputPath "C:\imgs" -OutputPath "C:\out" -Recurse -WhatIf

Notes:
- Requires ImageMagick CLI on PATH as `magick` (or pass -MagickPath to magick.exe).
- For HEIC input, Windows “HEIF Image Extensions” must be installed.
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [Parameter(Mandatory=$true)]
    [string]$InputPath,

    [Parameter(Mandatory=$true)]
    [string]$OutputPath,

    [switch]$Recurse,

    # If magick isn't on PATH, set this to the full path of magick.exe
    [string]$MagickPath = "C:\Program Files\ImageMagick-6.9.13-Q16-HDRI\convert.exe",

    # Tuning knobs (good defaults for printed diagrams)
    [string]$ContrastStretch = "0.5%x0.5%",
    [string]$Unsharp = "0x1.0+0.8+0.02",

    # Output format (png recommended for binary/line art)
    [ValidateSet("png","jpg","tif")]
    [string]$OutFormat = "png"
)

# --- Validate magick ---
try {
    & $MagickPath -version *> $null
} catch {
    throw "Could not run ImageMagick. Try setting -MagickPath to your magick.exe, e.g. 'C:\Program Files\ImageMagick-7.x.x-Q16-HDRI\magick.exe'"
}

# --- Normalize paths ---
$InputPath  = (Resolve-Path $InputPath).Path

# Create output directory if it doesn't exist (including parent directories)
if (-not (Test-Path $OutputPath)) {
    Write-Host "Creating output directory: $OutputPath"
    $null = New-Item -ItemType Directory -Path $OutputPath -Force
}
$OutputPath = (Resolve-Path $OutputPath).Path

# --- Collect images ---
$exts = @(".png",".jpg",".jpeg",".bmp",".tif",".tiff",".heic",".webp")
$gciParams = @{
    Path  = $InputPath
    File  = $true
}
if ($Recurse) { $gciParams.Recurse = $true }

$files = Get-ChildItem @gciParams | Where-Object { $exts -contains $_.Extension.ToLowerInvariant() }

if (-not $files) {
    Write-Host "No supported images found in: $InputPath"
    return
}

# Precompute max status line (so the single-line refresh always overwrites)
$maxLine = ("Processing {0} of {1}: " -f $files.Count, $files.Count) + ("X" * 120)
$maxLen  = $maxLine.Length

$idx = 0
foreach ($f in $files) {
    $idx++

    # Preserve folder structure under output
    $rel = $f.FullName.Substring($InputPath.Length).TrimStart('\')
    $outDir = Join-Path $OutputPath (Split-Path $rel -Parent)
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $outName = [IO.Path]::GetFileNameWithoutExtension($f.Name) + "." + $OutFormat
    $outFile = Join-Path $outDir $outName

    # Status: single refreshed line, padded to max length
    $status = ("Processing {0} of {1}: {2}" -f $idx, $files.Count, $rel)
    $status = $status.PadRight($maxLen)
    Write-Host "`r$status" -NoNewline

    $args = @(
        $f.FullName
        "-colorspace","Gray"
        "-contrast-stretch",$ContrastStretch
        "-unsharp",$Unsharp
        $outFile
    )

    if ($PSCmdlet.ShouldProcess($f.FullName, "Convert -> $outFile")) {
        & $MagickPath @args | Out-Null

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Warning "ImageMagick failed for: $($f.FullName)"
        }
    }
}

# Finish line cleanly
Write-Host "`rDone. Output in: $OutputPath".PadRight($maxLen)
Write-Host ""
Write-Host "Processed $($files.Count) images." -ForegroundColor Green