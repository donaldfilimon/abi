#Requires -Version 7.0
<#
.SYNOPSIS
    ABI Build Script for Windows/PowerShell

.DESCRIPTION
    Build script for the ABI Zig framework on Windows.
    Provides commands for building, testing, linking, and bootstrapping.

.EXAMPLE
    .\build.ps1                    # Build the library
    .\build.ps1 cli                # Build CLI binary
    .\build.ps1 test               # Run tests
    .\build.ps1 -Link             # Link Zig to PATH
    .\build.ps1 -Bootstrap        # Full setup

.PARAMETER Command
    Build command: test, cli, lib, mcp, lint, fix, check, check-parity

.PARAMETER Link
    Install and link Zig + ZLS to ~/.local/bin

.PARAMETER Bootstrap
    Full setup: install Zig + link + build

.PARAMETER Status
    Show Zig toolchain status

.PARAMETER Help
    Show this help message

.PARAMETER FeatureFlags
    Additional Zig feature flags (e.g., -Dfeat-gpu=false)
#>

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs,

    [Parameter(ParameterSetName="Link")]
    [switch]$Link,

    [Parameter(ParameterSetName="Bootstrap")]
    [switch]$Bootstrap,

    [Parameter(ParameterSetName="Status")]
    [switch]$Status,

    [Parameter(ParameterSetName="Help")]
    [switch]$Help,

    [Parameter(ParameterSetName="Command")]
    [ValidateSet("test", "cli", "lib", "mcp", "lint", "fix", "check", "check-parity")]
    [string]$Command
)

$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = (Get-Location).Path
}

function Show-Help {
    @"
Usage: .\build.ps1 [command] [options]

Commands:
    (default)         Build the library
    test              Run all tests
    cli               Build CLI binary
    lib               Build static library
    mcp               Build MCP server
    lint              Check formatting
    fix               Auto-format
    check             Full gate (lint + test + parity)
    check-parity      Verify mod/stub declaration parity
    --link            Install and link Zig + ZLS to ~/.local/bin
    --status          Show Zig toolchain status
    --bootstrap       Full setup: install Zig + link + build
    --help            Show this help

Options:
    -Dfeat-*          Feature flags (e.g., -Dfeat-gpu=false)
    -Dgpu-backend=*   GPU backend (metal, cuda, vulkan, etc.)

Examples:
    .\build.ps1                    # Build library
    .\build.ps1 cli                # Build CLI
    .\build.ps1 test               # Run tests
    .\build.ps1 --link             # Link Zig to PATH
    .\build.ps1 --bootstrap        # Full setup

Cross-compilation (from Linux/macOS/WSL):
    .\tools\crossbuild.ps1 windows  # Build for Windows

"@
    exit 0
}

function Find-Zig {
    $candidates = @(
        "$env:USERPROFILE\.zvm\bin\zig.exe",
        "$env:LOCALAPPDATA\Programs\Zig\zig.exe",
        "$env:USERPROFILE\.local\bin\zig.exe",
        "$env:USERPROFILE\.zigly\versions\*\zig.exe",
        (Get-Command zig -ErrorAction SilentlyContinue).Source
    )

    foreach ($candidate in $candidates) {
        if ($candidate -match '\*') {
            $matches = Get-ChildItem $candidate -ErrorAction SilentlyContinue
            if ($matches) {
                $latest = $matches | Sort-Object Name -Descending | Select-Object -First 1
                if ($latest.FullName) { return $latest.FullName }
            }
        } elseif ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

# Main logic
if ($Help) {
    Show-Help
}

if ($Status) {
    $zig = Find-Zig
    if ($zig) {
        Write-Host "Zig found: $zig" -ForegroundColor Green
        & $zig version
    } else {
        Write-Host "Zig not found. Run '.\build.ps1 -Bootstrap' or 'tools\zigly --bootstrap'" -ForegroundColor Yellow
    }
    exit 0
}

$zig = Find-Zig
if (-not $zig) {
    Write-Host "Error: Zig not found. Run '.\build.ps1 -Bootstrap' or 'tools\zigly --bootstrap'" -ForegroundColor Red
    exit 1
}

Write-Host "Using Zig: $zig" -ForegroundColor Cyan
Write-Host ""

if ($Link) {
    Write-Host "Linking Zig + ZLS to PATH..." -ForegroundColor Cyan
    
    $zigDir = Split-Path $zig -Parent
    $zigLibDir = Join-Path (Split-Path $zigDir -Parent) "lib"
    
    # Add to PATH if not already there
    if ($env:PATH -notlike "*$zigDir*") {
        $env:PATH = "$zigDir;$env:PATH"
        [Environment]::SetEnvironmentVariable("PATH", "$env:PATH", "User")
    }
    
    Write-Host "Zig is now available in your PATH" -ForegroundColor Green
    exit 0
}

if ($Bootstrap) {
    Write-Host "Bootstrapping: installing and linking Zig + ZLS..." -ForegroundColor Cyan
    
    # Try to use zigly if available
    $ziglyPath = Join-Path $ScriptDir "tools\zigly"
    if (Test-Path $ziglyPath) {
        & $ziglyPath --bootstrap
    } else {
        Write-Host "Zigly not found. Please install Zig manually." -ForegroundColor Yellow
        Write-Host "Download from: https://ziglang.org/download/" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "`nBuilding project..." -ForegroundColor Cyan
}

# Determine command
$buildCmd = if ($Command) { $Command } else { "lib" }

# Build
$args = @($buildCmd) + $RemainingArgs
Write-Host "Running: zig build $($args -join ' ')" -ForegroundColor Gray
& $zig build @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nBuild completed successfully!" -ForegroundColor Green