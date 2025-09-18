# Abi AI Framework Release Script for Windows
# This script builds and packages the Abi AI Framework for release

param(
    [switch]$SkipTests,
    [switch]$SkipArchive,
    [switch]$Verbose
)

# Colors for output
$RED = "Red"
$GREEN = "Green"
$YELLOW = "Yellow"
$BLUE = "Cyan"
$WHITE = "White"

function Write-Status {
    param([string]$Message)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] " -ForegroundColor $BLUE -NoNewline
    Write-Host $Message -ForegroundColor $WHITE
}

function Write-Success {
    param([string]$Message)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] " -ForegroundColor $GREEN -NoNewline
    Write-Host $Message -ForegroundColor $WHITE
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] " -ForegroundColor $YELLOW -NoNewline
    Write-Host $Message -ForegroundColor $WHITE
}

function Write-Error {
    param([string]$Message)
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] " -ForegroundColor $RED -NoNewline
    Write-Host $Message -ForegroundColor $WHITE
}

Write-Host "🚀 Abi AI Framework Release Build Script (Windows)" -ForegroundColor $GREEN
Write-Host "=================================================" -ForegroundColor $GREEN
Write-Host ""

# Check prerequisites
function Test-Prerequisites {
    Write-Status "Checking prerequisites..."

    try {
        $zigVersion = & zig version 2>$null
        if ($LASTEXITCODE -ne 0) { throw "Zig not found" }
        Write-Success "Zig version: $zigVersion"
    } catch {
        Write-Error "Zig is not installed or not in PATH"
        exit 1
    }

    try {
        $gitVersion = & git --version 2>$null
        if ($LASTEXITCODE -ne 0) { throw "Git not found" }
        Write-Success "Git found: $($gitVersion -replace 'git version ')"
    } catch {
        Write-Error "Git is not installed or not in PATH"
        exit 1
    }

    Write-Success "Prerequisites check passed"
}

# Clean build artifacts
function Clear-BuildArtifacts {
    Write-Status "Cleaning previous build artifacts..."
    if (Test-Path "zig-cache") { Remove-Item -Recurse -Force "zig-cache" }
    if (Test-Path "zig-out") { Remove-Item -Recurse -Force "zig-out" }
    Write-Success "Build artifacts cleaned"
}

# Run tests
function Invoke-Tests {
    if ($SkipTests) {
        Write-Warning "Skipping tests as requested"
        return
    }

    Write-Status "Running comprehensive test suite..."

    try {
        & zig build test
        if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
        Write-Success "All tests passed"
    } catch {
        Write-Error "Tests failed!"
        exit 1
    }
}

# Build release binaries
function Build-Release {
    Write-Status "Building release binaries..."

    try {
        & zig build -Doptimize=ReleaseFast
        if ($LASTEXITCODE -ne 0) { throw "Release build failed" }
        Write-Success "Release binaries built successfully"
    } catch {
        Write-Error "Release build failed!"
        exit 1
    }

    # List built artifacts
    Write-Status "Built artifacts:"
    if (Test-Path "zig-out\bin") {
        Get-ChildItem "zig-out\bin" -File | ForEach-Object {
            $size = [math]::Round($_.Length / 1KB, 1)
            Write-Host "  $($_.FullName) (${size}KB)" -ForegroundColor $WHITE
        }
    } else {
        Write-Warning "No binaries found in zig-out\bin"
    }
}

# Generate documentation
function New-Documentation {
    Write-Status "Generating API documentation..."

    try {
        & zig build docs
        if ($LASTEXITCODE -eq 0) {
            Write-Success "API documentation generated"
        } else {
            Write-Warning "Documentation generation failed, but continuing..."
        }
    } catch {
        Write-Warning "Documentation generation failed, but continuing..."
    }
}

# Create release archive
function New-ReleaseArchive {
    if ($SkipArchive) {
        Write-Warning "Skipping archive creation as requested"
        return
    }

    try {
        $version = & git describe --tags --abbrev=0 2>$null
        if ($LASTEXITCODE -ne 0) { $version = "v1.0.0" }
    } catch {
        $version = "v1.0.0"
    }

    $archiveName = "abi-framework-$($version -replace '^v')-windows-$($env:PROCESSOR_ARCHITECTURE.ToLower())"
    Write-Status "Creating release archive: $archiveName.zip"

    # Create temporary directory for archive
    $tempDir = [System.IO.Path]::GetTempPath() + [System.Guid]::NewGuid().ToString()
    $archiveDir = Join-Path $tempDir $archiveName

    New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null

    # Copy built artifacts
    if (Test-Path "zig-out") { Copy-Item -Recurse "zig-out" $archiveDir }
    if (Test-Path "docs") { Copy-Item -Recurse "docs" $archiveDir }
    if (Test-Path "README.md") { Copy-Item "README.md" $archiveDir }
    if (Test-Path "CHANGELOG.md") { Copy-Item "CHANGELOG.md" $archiveDir }
    if (Test-Path "LICENSE") { Copy-Item "LICENSE" $archiveDir }

    # Create ZIP archive
    $zipPath = "$archiveName.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath }

    Add-Type -AssemblyName "System.IO.Compression.FileSystem"
    [System.IO.Compression.ZipFile]::CreateFromDirectory($archiveDir, $zipPath)

    # Cleanup
    Remove-Item -Recurse -Force $tempDir

    Write-Success "Release archive created: $zipPath"
}

# Show release information
function Show-ReleaseInfo {
    try {
        $version = & git describe --tags --abbrev=0 2>$null
        if ($LASTEXITCODE -ne 0) { $version = "v1.0.0" }
    } catch {
        $version = "v1.0.0"
    }

    Write-Host ""
    Write-Host "🎉 Release Build Complete!" -ForegroundColor $GREEN
    Write-Host "==========================" -ForegroundColor $GREEN
    Write-Host "Version: $version" -ForegroundColor $WHITE
    Write-Host "Date: $(Get-Date)" -ForegroundColor $WHITE
    Write-Host ""

    Write-Host "Built Artifacts:" -ForegroundColor $WHITE
    if (Test-Path "zig-out\bin") {
        Get-ChildItem "zig-out\bin" -File | ForEach-Object {
            Write-Host "  $($_.Name)" -ForegroundColor $WHITE
        }
    } else {
        Write-Host "  No binaries found" -ForegroundColor $YELLOW
    }
    Write-Host ""

    Write-Host "Release Archive:" -ForegroundColor $WHITE
    Get-ChildItem "*.zip" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  $($_.Name) ($([math]::Round($_.Length / 1MB, 1))MB)" -ForegroundColor $WHITE
    }
    if (!(Get-ChildItem "*.zip" -ErrorAction SilentlyContinue)) {
        Write-Host "  No archive created" -ForegroundColor $YELLOW
    }
    Write-Host ""

    Write-Host "Next Steps:" -ForegroundColor $BLUE
    Write-Host "1. Test the built binaries: .\zig-out\bin\abi.exe --help" -ForegroundColor $WHITE
    Write-Host "2. Run benchmarks: zig build benchmark" -ForegroundColor $WHITE
    Write-Host "3. Deploy using: See deploy\ directory for Kubernetes manifests" -ForegroundColor $WHITE
    Write-Host "4. Publish documentation: See docs\ directory" -ForegroundColor $WHITE
    Write-Host "5. Push to repository: git push && git push --tags" -ForegroundColor $WHITE
}

# Main execution
function Main {
    Write-Status "Starting Abi AI Framework release build..."

    Test-Prerequisites
    Clear-BuildArtifacts
    Invoke-Tests
    Build-Release
    New-Documentation
    New-ReleaseArchive
    Show-ReleaseInfo

    Write-Host ""
    Write-Success "🎉 Abi AI Framework release build completed successfully!"
    Write-Success "Ready for production deployment!"
}

# Run main function
Main
