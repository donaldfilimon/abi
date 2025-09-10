param(
  [string]$InstallRoot = ".tools/zig-0.15.1"
)

$ErrorActionPreference = 'Stop'

Write-Host "Setting up Zig 0.15.1 in $InstallRoot"

$ver = '0.15.1'

$index = Invoke-WebRequest -Uri 'https://ziglang.org/download/index.json' -UseBasicParsing
$data = $index.Content | ConvertFrom-Json
$url = $data.$ver.'x86_64-windows'.tarball

if (-not (Test-Path $InstallRoot)) {
  New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
}

$zip = Join-Path $InstallRoot "zig-$ver.zip"
Write-Host "Downloading from $url"
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing

Write-Host "Extracting to $InstallRoot"
Expand-Archive -Force -LiteralPath $zip -DestinationPath $InstallRoot
Remove-Item $zip -Force

$folder = Get-ChildItem $InstallRoot -Directory | Where-Object { $_.Name -like "zig-*" } | Select-Object -First 1
if (-not $folder) { throw "Failed to extract Zig archive." }

$zig = Join-Path $folder.FullName 'zig.exe'
& $zig version

Write-Host "Zig installed at: $zig"

