$im = "C:\Program Files\ImageMagick-6.9.13-Q16-HDRI\convert.exe"
$source = "C:\Users\mortenha\source\repos\Chess\ChessFEN\data\raw\batch_005"
$dest = "C:\Users\mortenha\source\repos\Chess\ChessFEN\data\raw\batch_005"

New-Item -ItemType Directory -Path $dest -Force | Out-Null

$images = Get-ChildItem $source -Filter *.heic
Write-Host "Found $($images.Count) images at source: $source" -ForegroundColor Cyan

$count = 0
$images | ForEach-Object {
    $count++
    $outPath = Join-Path $dest ($_.BaseName + ".jpg")
    Write-Host "[$count/$($images.Count)] Converting: $($_.Name) -> $($_.BaseName).jpg" -ForegroundColor Yellow
    & $im $_.FullName $outPath
}

Write-Host "`nDone! Processed $count images to: $dest" -ForegroundColor Green
