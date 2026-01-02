<#
Enhance and augment chess tile images using ImageMagick.

Pipeline:
  1. ENHANCE: Grayscale + Contrast stretch + Sharpen (same as copyAndMagickify.ps1)
  2. AUGMENT: Create variations (blur, rotation, skew, brightness, noise, scale)

Augmentations applied:
  - Slight rotations (-3° to +3°)
  - Blur variations (gaussian blur)
  - Skew/shear distortions
  - Brightness/contrast variations
  - Noise addition
  - Slight scaling variations

Usage:
  .\augmentTiles.ps1 -InputPath "data\tiles\batch_007" -OutputPath "data\tiles\batch_007_augmented" -Recurse
  .\augmentTiles.ps1 -InputPath "data\tiles\batch_007" -OutputPath "data\tiles\batch_007_augmented" -Recurse -VariationsPerImage 5
#>

[CmdletBinding(SupportsShouldProcess=$true)]
param(
    [Parameter(Mandatory=$true)]
    [string]$InputPath,

    [Parameter(Mandatory=$true)]
    [string]$OutputPath,

    [switch]$Recurse,

    # Number of augmented variations per input image
    [int]$VariationsPerImage = 8,

    # Path to ImageMagick convert
    [string]$MagickPath = "C:\Program Files\ImageMagick-6.9.13-Q16-HDRI\convert.exe",

    # Enhancement settings (from copyAndMagickify.ps1)
    [string]$ContrastStretch = "0.5%x0.5%",
    [string]$Unsharp = "0x1.0+0.8+0.02",

    # Also save the enhanced original (no augmentation)
    [switch]$IncludeOriginal = $true
)

# --- Validate magick ---
try {
    & $MagickPath -version *> $null
} catch {
    throw "Could not run ImageMagick. Set -MagickPath to your convert.exe"
}

# --- Normalize paths ---
$InputPath = (Resolve-Path $InputPath).Path

if (-not (Test-Path $OutputPath)) {
    Write-Host "Creating output directory: $OutputPath"
    $null = New-Item -ItemType Directory -Path $OutputPath -Force
}
$OutputPath = (Resolve-Path $OutputPath).Path

# --- Gather input images ---
$extensions = @("*.png", "*.jpg", "*.jpeg", "*.bmp")
$files = @()
foreach ($ext in $extensions) {
    if ($Recurse) {
        $files += Get-ChildItem -Path $InputPath -Filter $ext -Recurse -File
    } else {
        $files += Get-ChildItem -Path $InputPath -Filter $ext -File
    }
}

if ($files.Count -eq 0) {
    Write-Warning "No image files found in $InputPath"
    exit
}

Write-Host "Found $($files.Count) images to augment"
Write-Host "Generating $VariationsPerImage variations each = $($files.Count * $VariationsPerImage) augmented images"
Write-Host ""

# --- Augmentation definitions ---
# Each augmentation returns an array of ImageMagick arguments
$augmentations = @(
    @{ Name = "rot_left"; Args = @("-rotate", "-3", "-background", "white", "-gravity", "center", "-extent", "80x80") },
    @{ Name = "rot_right"; Args = @("-rotate", "3", "-background", "white", "-gravity", "center", "-extent", "80x80") },
    @{ Name = "blur_light"; Args = @("-blur", "0x0.5") },
    @{ Name = "blur_med"; Args = @("-blur", "0x1.0") },
    @{ Name = "shear_x"; Args = @("-shear", "3x0", "-background", "white", "-gravity", "center", "-extent", "80x80") },
    @{ Name = "shear_y"; Args = @("-shear", "0x3", "-background", "white", "-gravity", "center", "-extent", "80x80") },
    @{ Name = "bright_up"; Args = @("-modulate", "110,100,100") },
    @{ Name = "bright_down"; Args = @("-modulate", "90,100,100") },
    @{ Name = "noise"; Args = @("-attenuate", "0.1", "+noise", "Gaussian") },
    @{ Name = "scale_up"; Args = @("-resize", "105%", "-gravity", "center", "-extent", "80x80") },
    @{ Name = "scale_down"; Args = @("-resize", "95%", "-gravity", "center", "-background", "white", "-extent", "80x80") },
    @{ Name = "rot_blur"; Args = @("-rotate", "2", "-blur", "0x0.5", "-background", "white", "-gravity", "center", "-extent", "80x80") }
)

# Select random subset if we want fewer variations
$selectedAugmentations = $augmentations | Get-Random -Count ([Math]::Min($VariationsPerImage, $augmentations.Count))

$processed = 0
$errors = 0

foreach ($file in $files) {
    # Preserve directory structure
    $relativePath = $file.FullName.Substring($InputPath.Length).TrimStart('\', '/')
    $relativeDir = Split-Path $relativePath -Parent
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    
    $outDir = Join-Path $OutputPath $relativeDir
    if (-not (Test-Path $outDir)) {
        $null = New-Item -ItemType Directory -Path $outDir -Force
    }
    
    # Optionally save enhanced original (grayscale + contrast + sharpen, no augmentation)
    if ($IncludeOriginal) {
        $origOut = Join-Path $outDir "$baseName`_orig.png"
        $origArgs = @(
            $file.FullName,
            "-colorspace", "Gray",
            "-contrast-stretch", $ContrastStretch,
            "-unsharp", $Unsharp,
            $origOut
        )
        & $MagickPath @origArgs 2>$null
    }
    
    # Apply each selected augmentation (enhancement + augmentation)
    foreach ($aug in $selectedAugmentations) {
        $outFile = Join-Path $outDir "$baseName`_$($aug.Name).png"
        
        # Build full command: enhance first, then augment
        $fullArgs = @($file.FullName)
        $fullArgs += @("-colorspace", "Gray")
        $fullArgs += @("-contrast-stretch", $ContrastStretch)
        $fullArgs += @("-unsharp", $Unsharp)
        $fullArgs += $aug.Args
        $fullArgs += @($outFile)
        
        & $MagickPath @fullArgs 2>$null
        if ($LASTEXITCODE -ne 0) {
            $errors++
        }
    }
    
    $processed++
    if ($processed % 50 -eq 0) {
        Write-Host "  Processed $processed / $($files.Count) images..."
    }
}

Write-Host ""
Write-Host "Done!"
Write-Host "  Input images:  $($files.Count)"
Write-Host "  Augmentations: $($selectedAugmentations.Count) per image"
Write-Host "  Total output:  ~$($files.Count * ($selectedAugmentations.Count + $(if($IncludeOriginal){1}else{0}))) images"
Write-Host "  Output folder: $OutputPath"
if ($errors -gt 0) {
    Write-Warning "  Errors: $errors"
}
