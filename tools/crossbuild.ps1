#Requires -Version 7.0
<#
.SYNOPSIS
    ABI Cross-Platform Build Tool for Windows

.DESCRIPTION
    Builds ABI for multiple platforms from Windows.
    Use from WSL or when cross-compiling.

.PARAMETER Platform
    Target platform: linux, macos, windows, wasm, ios, android, freebsd, all

.PARAMETER Optimize
    Build optimization: Debug, ReleaseSafe, ReleaseFast (default: ReleaseSafe)

.PARAMETER List
    List available targets

.PARAMETER Clean
    Clean all build artifacts

.EXAMPLE
    .\crossbuild.ps1 --all         # Build all platforms
    .\crossbuild.ps1 windows       # Build for Windows
    .\crossbuild.ps1 --list       # List available targets
#>

param(
    [Parameter(Position=0)]
    [ValidateSet("linux", "macos", "windows", "wasm", "ios", "android", "freebsd", "all", "list", "clean")]
    [string]$Platform = "list",

    [ValidateSet("Debug", "ReleaseSafe", "ReleaseFast")]
    [string]$Optimize = "ReleaseSafe",

    [switch]$List,

    [switch]$Clean,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$FeatureFlags
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = (Get-Location).Path
}
$RepoRoot = (Get-Item $ScriptDir).Parent.FullName
$OutBase = Join-Path $RepoRoot "zig-out"

function Show-List {
    Write-Host "Available cross-build targets:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Platform   Targets"
    Write-Host "  --------   -------"
    Write-Host "  linux      aarch64-linux-musl, x86_64-linux-musl"
    Write-Host "  macos      aarch64-macos, x86_64-macos"
    Write-Host "  windows    x86_64-windows-gnu"
    Write-Host "  wasm       wasm32-wasi"
    Write-Host "  ios        aarch64-ios"
    Write-Host "  android    aarch64-linux-android"
    Write-Host "  freebsd    x86_64-freebsd, aarch64-freebsd"
    Write-Host ""
    Write-Host "Usage: .\crossbuild.ps1 <platform> [-Optimize ReleaseSafe|ReleaseFast]" -ForegroundColor Gray
    exit 0
}

function Find-Zig {
    $candidates = @(
        "$env:USERPROFILE\.zvm\bin\zig.exe",
        "$env:LOCALAPPDATA\Programs\Zig\zig.exe",
        "$env:USERPROFILE\.local\bin\zig.exe",
        (Get-Command zig -ErrorAction SilentlyContinue).Source
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

function Build-Target {
    param(
        [string]$Target,
        [string]$Flags = ""
    )

    Write-Host "Building $Target (optimize=$Optimize)..." -ForegroundColor Yellow

    $zig = Find-Zig
    if (-not $zig) {
        Write-Host "Zig not found. Run '.\build.ps1 -Bootstrap'" -ForegroundColor Red
        exit 1
    }

    $zigDir = Split-Path $zig -Parent
    $zigLibDir = Join-Path (Split-Path $zigDir -Parent) "lib"

    $args = @(
        "lib",
        "--zig-lib-dir", $zigLibDir,
        "--global-cache-dir", "$env:USERPROFILE\.cache\zig",
        "--cache-dir", "$ScriptDir\.zig-cache",
        "-Dtarget=$Target",
        "-Doptimize=$Optimize"
    )

    if ($Flags) {
        $args += $Flags.Split(" ", [StringSplitOptions]::RemoveEmptyEntries)
    }

    $outDir = Join-Path $OutBase $Target
    $args += @("--prefix", $outDir)

    & $zig @args

    Write-Host "  $Target -> $outDir" -ForegroundColor Green
}

function Get-FeatureFlags {
    param([string]$Platform)

    switch ($Platform) {
        "wasm"  { return "-Dfeat-gpu=false -Dfeat-database=false -Dfeat-network=false -Dfeat-observability=false -Dfeat-web=false -Dfeat-pages=false -Dfeat-cloud=false -Dfeat-storage=false -Dfeat-compute=false -Dfeat-desktop=false -Dfeat-lsp=false -Dfeat-mcp=false" }
        "ios"   { return "-Dfeat-mobile=true -Dgpu-backend=metal,opengles" }
        "android" { return "-Dfeat-mobile=true -Dgpu-backend=vulkan,opengles" }
        "windows" { return "-Dgpu-backend=cuda,vulkan,opengl,opengles,stdgpu" }
        default { return "" }
    }
}

# Main logic
if ($List -or $Platform -eq "list") {
    Show-List
}

if ($Clean) {
    Write-Host "Cleaning all build artifacts..." -ForegroundColor Yellow
    Remove-Item -Path $OutBase -Recurse -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $RepoRoot ".zig-cache") -Recurse -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

$flags = (Get-FeatureFlags -Platform $Platform) + " " + ($FeatureFlags -join " ")

switch ($Platform) {
    "all" {
        Write-Host "=== Building all platforms ===" -ForegroundColor Cyan
        Build-Target -Target "aarch64-linux-musl" -Flags $flags
        Build-Target -Target "x86_64-linux-musl" -Flags $flags
        Build-Target -Target "aarch64-macos" -Flags $flags
        Build-Target -Target "x86_64-macos" -Flags $flags
        Build-Target -Target "x86_64-windows-gnu" -Flags "-Dgpu-backend=cuda,vulkan,opengl,opengles,stdgpu"
        Build-Target -Target "wasm32-wasi" -Flags "-Dfeat-gpu=false -Dfeat-database=false -Dfeat-network=false -Dfeat-observability=false -Dfeat-web=false -Dfeat-pages=false -Dfeat-cloud=false -Dfeat-storage=false -Dfeat-compute=false -Dfeat-desktop=false -Dfeat-lsp=false -Dfeat-mcp=false"
        Build-Target -Target "aarch64-ios" -Flags "-Dfeat-mobile=true -Dgpu-backend=metal,opengles"
        Build-Target -Target "aarch64-linux-android" -Flags "-Dfeat-mobile=true -Dgpu-backend=vulkan,opengles"
        Build-Target -Target "x86_64-freebsd" -Flags $flags
        Build-Target -Target "aarch64-freebsd" -Flags $flags
    }
    "linux" {
        Write-Host "=== Linux ===" -ForegroundColor Cyan
        Build-Target -Target "aarch64-linux-musl" -Flags $flags
        Build-Target -Target "x86_64-linux-musl" -Flags $flags
    }
    "macos" {
        Write-Host "=== macOS ===" -ForegroundColor Cyan
        Build-Target -Target "aarch64-macos" -Flags $flags
        Build-Target -Target "x86_64-macos" -Flags $flags
    }
    "windows" {
        Write-Host "=== Windows ===" -ForegroundColor Cyan
        Build-Target -Target "x86_64-windows-gnu" -Flags "-Dgpu-backend=cuda,vulkan,opengl,opengles,stdgpu"
    }
    "wasm" {
        Write-Host "=== WASM/WASI ===" -ForegroundColor Cyan
        Build-Target -Target "wasm32-wasi" -Flags "-Dfeat-gpu=false -Dfeat-database=false -Dfeat-network=false -Dfeat-observability=false -Dfeat-web=false -Dfeat-pages=false -Dfeat-cloud=false -Dfeat-storage=false -Dfeat-compute=false -Dfeat-desktop=false -Dfeat-lsp=false -Dfeat-mcp=false"
    }
    "ios" {
        Write-Host "=== iOS ===" -ForegroundColor Cyan
        Build-Target -Target "aarch64-ios" -Flags "-Dfeat-mobile=true -Dgpu-backend=metal,opengles"
    }
    "android" {
        Write-Host "=== Android ===" -ForegroundColor Cyan
        Build-Target -Target "aarch64-linux-android" -Flags "-Dfeat-mobile=true -Dgpu-backend=vulkan,opengles"
    }
    "freebsd" {
        Write-Host "=== FreeBSD ===" -ForegroundColor Cyan
        Build-Target -Target "x86_64-freebsd" -Flags $flags
        Build-Target -Target "aarch64-freebsd" -Flags $flags
    }
}

Write-Host "`n=== Cross-Build Complete ===" -ForegroundColor Green