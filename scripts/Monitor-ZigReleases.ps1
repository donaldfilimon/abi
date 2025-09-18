# Zig Release Monitor PowerShell Script
# Monitors Zig releases and updates project configuration accordingly

param(
    [switch]$UpdateProject,
    [switch]$TestBuild
)

Write-Host "🔍 Zig Release Monitor" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan

# Configuration
$GitHubApiUrl = "https://api.github.com/repos/ziglang/zig/releases"
$CurrentVersionFile = ".zigversion"
$CiWorkflowFile = ".github/workflows/ci.yml"
$DocsWorkflowFile = ".github/workflows/deploy_docs.yml"

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Get-LatestReleases {
    Write-Status "Fetching latest Zig releases from GitHub..."

    try {
        $response = Invoke-RestMethod -Uri $GitHubApiUrl -Method Get
        $response | ConvertTo-Json -Depth 10 | Out-File -FilePath "$env:TEMP\zig_releases.json" -Encoding UTF8
        Write-Success "Successfully fetched release information"
        return $response
    }
    catch {
        Write-Error "Failed to fetch releases from GitHub API: $($_.Exception.Message)"
        return $null
    }
}

function Get-CurrentVersion {
    if (Test-Path $CurrentVersionFile) {
        $version = Get-Content $CurrentVersionFile -Raw
        Write-Status "Current project version: $($version.Trim())"
        return $version.Trim()
    } else {
        Write-Warning "No current version file found"
        return $null
    }
}

function Update-ProjectFiles {
    param([string]$NewVersion, [bool]$IsStable)

    Write-Status "Checking if project needs updates..."

    $currentVersion = Get-CurrentVersion

    if ($currentVersion -eq $NewVersion) {
        Write-Success "Project is already up to date"
        return $false
    }

    Write-Warning "Project version mismatch detected"
    Write-Status "Current: $currentVersion"
    Write-Status "Latest: $NewVersion"

    # Update .zigversion file
    $NewVersion | Out-File -FilePath $CurrentVersionFile -Encoding UTF8
    Write-Success "Updated $CurrentVersionFile"

    # Update CI workflow
    if (Test-Path $CiWorkflowFile) {
        $content = Get-Content $CiWorkflowFile -Raw
        # Add new version to matrix
        $content = $content -replace "0\.16\.0-dev\.\d+", "`$&,$NewVersion"
        $content | Out-File -FilePath $CiWorkflowFile -Encoding UTF8
        Write-Success "Updated CI workflow with new version"
    }

    # Update docs workflow
    if (Test-Path $DocsWorkflowFile) {
        $content = Get-Content $DocsWorkflowFile -Raw
        $content = $content -replace "version: 0\.16\.0-dev\.\d+", "version: $NewVersion"
        $content | Out-File -FilePath $DocsWorkflowFile -Encoding UTF8
        Write-Success "Updated docs workflow with new version"
    }

    # Create update summary
    $summary = @"
# Zig Version Update - $(Get-Date)

## Changes Made
- Updated $CurrentVersionFile to $NewVersion
- Updated CI pipeline to include $NewVersion
- Updated documentation workflow to use $NewVersion

## Version Details
- Previous version: $currentVersion
- New version: $NewVersion
- Stable release: $(if ($IsStable) { "Yes" } else { "No" })
"@

    $summary | Out-File -FilePath "$env:TEMP\zig_update_summary.md" -Encoding UTF8
    Write-Success "Update summary created"

    return $true
}

function Test-ZigBuild {
    param([string]$Version)

    Write-Status "Testing build with Zig $Version..."

    try {
        $zigVersion = & zig version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Zig version: $zigVersion"
        } else {
            Write-Warning "Zig not found or failed to get version"
            return $true  # Continue even if zig isn't available
        }
    }
    catch {
        Write-Warning "Zig command failed, skipping version check"
        return $true
    }

    # Test project build
    try {
        & zig build
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Project builds successfully with $Version"
            return $true
        } else {
            Write-Error "Project build failed with $Version"
            return $false
        }
    }
    catch {
        Write-Error "Build test failed: $($_.Exception.Message)"
        return $false
    }
}

# Main execution
function Main {
    $releases = Get-LatestReleases
    if (-not $releases) {
        exit 1
    }

    # Find latest stable release
    $latestStable = $releases | Where-Object { -not $_.prerelease } | Select-Object -First 1
    $latestDev = $releases | Select-Object -First 1

    Write-Success "Latest stable: $(if ($latestStable) { $latestStable.tag_name } else { 'None' })"
    Write-Success "Latest dev: $($latestDev.tag_name)"

    $updated = $false

    # Prefer stable release if available
    if ($latestStable) {
        Write-Status "Stable release available: $($latestStable.tag_name)"
        if ($UpdateProject) {
            $updated = Update-ProjectFiles -NewVersion $latestStable.tag_name -IsStable $true
        }
        if ($TestBuild) {
            Test-ZigBuild -Version $latestStable.tag_name
        }
    } else {
        Write-Status "No stable release, using latest dev: $($latestDev.tag_name)"
        if ($UpdateProject) {
            $updated = Update-ProjectFiles -NewVersion $latestDev.tag_name -IsStable $false
        }
        if ($TestBuild) {
            Test-ZigBuild -Version $latestDev.tag_name
        }
    }

    # Show summary if updated
    $summaryFile = "$env:TEMP\zig_update_summary.md"
    if ((Test-Path $summaryFile) -and $updated) {
        Write-Host ""
        Write-Host "📋 Update Summary:" -ForegroundColor Cyan
        Write-Host "==================" -ForegroundColor Cyan
        Get-Content $summaryFile
    }

    Write-Success "Zig release monitoring completed"
}

# Run main function
Main
