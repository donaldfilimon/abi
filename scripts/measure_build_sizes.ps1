# PowerShell version of build size measurement script
# Compares different optimization modes and feature configurations

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "ABI Build Size Comparison" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Clean previous builds
if (Test-Path "zig-out") { Remove-Item -Recurse -Force "zig-out" }
if (Test-Path "zig-cache") { Remove-Item -Recurse -Force "zig-cache" }

# Function to build and measure
function Measure-Build {
    param(
        [string]$Name,
        [string]$Flags
    )
    
    Write-Host "Building: $Name" -ForegroundColor Yellow
    Write-Host "Flags: $Flags" -ForegroundColor Gray
    
    # Build
    try {
        if ($Flags) {
            Invoke-Expression "zig build $Flags" 2>&1 | Out-Null
        } else {
            zig build 2>&1 | Out-Null
        }
    } catch {
        Write-Host "  ❌ Build failed" -ForegroundColor Red
        Write-Host ""
        return
    }
    
    # Measure size
    $binPath = "zig-out\bin\abi.exe"
    if (Test-Path $binPath) {
        $size = (Get-Item $binPath).Length
        $sizeKB = [math]::Round($size / 1KB, 2)
        $sizeMB = [math]::Round($size / 1MB, 2)
        Write-Host "  ✅ Size: $sizeKB KB ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Binary not found" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Baseline - Debug build
Measure-Build "Debug (baseline)" ""

# Release builds
Measure-Build "ReleaseSafe" "-Doptimize=ReleaseSafe"
Measure-Build "ReleaseFast" "-Doptimize=ReleaseFast"
Measure-Build "ReleaseSmall" "-Doptimize=ReleaseSmall"

# Minimal build (no optional features)
Measure-Build "Minimal (ReleaseSmall, no AI/GPU/Web)" `
    "-Doptimize=ReleaseSmall -Denable-ai=false -Denable-gpu=false -Denable-web=false -Denable-monitoring=false"

# Database-only build
Measure-Build "Database-only (ReleaseSmall)" `
    "-Doptimize=ReleaseSmall -Denable-ai=false -Denable-gpu=false -Denable-web=false -Denable-monitoring=false"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Build size comparison complete!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Recommendations:" -ForegroundColor Yellow
Write-Host "  - For production: zig build -Doptimize=ReleaseSafe" -ForegroundColor White
Write-Host "  - For max speed: zig build -Doptimize=ReleaseFast" -ForegroundColor White
Write-Host "  - For min size: zig build -Doptimize=ReleaseSmall" -ForegroundColor White
Write-Host "  - For embedded: Add feature flags to disable unused features" -ForegroundColor White
Write-Host ""
