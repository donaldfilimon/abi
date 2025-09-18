# Automated Performance Monitoring System for Windows
# Continuously monitors performance metrics and generates reports

param(
    [switch]$Continuous,
    [int]$IntervalMinutes = 60,
    [string]$OutputPath = ".\reports"
)

Write-Host "📊 Automated Performance Monitoring System" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$ReportsDir = Join-Path $ProjectRoot $OutputPath
$MetricsDir = Join-Path $ReportsDir "metrics"
$LogsDir = Join-Path $ReportsDir "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportFile = Join-Path $ReportsDir "performance_report_$Timestamp.md"

# Create directories
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $MetricsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

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

function Write-Metric {
    param([string]$Message)
    Write-Host "[METRIC] $Message" -ForegroundColor Magenta
}

# System information collection
function Collect-SystemInfo {
    Write-Status "Collecting system information..."

    $cpuName = (Get-WmiObject -Class Win32_Processor).Name
$cpuCores = (Get-WmiObject -Class Win32_Processor).NumberOfCores
$cpuSpeed = (Get-WmiObject -Class Win32_Processor).MaxClockSpeed
$totalMem = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
$freeMem = [math]::Round((Get-WmiObject -Class Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)

    $diskInfo = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 } | ForEach-Object {
        "$($_.DeviceID) - $([math]::Round($_.Size / 1GB, 2)) GB total, $([math]::Round($_.FreeSpace / 1GB, 2)) GB free"
    }

    $osCaption = (Get-WmiObject -Class Win32_OperatingSystem).Caption
    $osVersion = (Get-WmiObject -Class Win32_OperatingSystem).Version
    $lastBoot = (Get-WmiObject -Class Win32_OperatingSystem).LastBootUpTime

    $systemInfo = "System Information - $(Get-Date)`n" +
                  "================================`n`n" +
                  "CPU Information:`n" +
                  "$cpuName`n" +
                  "$cpuCores cores`n" +
                  "$cpuSpeed MHz`n`n" +
                  "Memory Information:`n" +
                  "$totalMem GB total`n" +
                  "$freeMem GB free`n`n" +
                  "Disk Information:`n" +
                  "$($diskInfo -join "`n")`n`n" +
                  "OS Information:`n" +
                  "$osCaption`n" +
                  "$osVersion`n`n" +
                  "Uptime:`n" +
                  "$lastBoot"

    $systemInfo | Out-File -FilePath (Join-Path $MetricsDir "system_info_$Timestamp.txt")
}

# Build performance monitoring
function Monitor-BuildPerformance {
    Write-Status "Monitoring build performance..."

    Push-Location $ProjectRoot

    try {
        $buildStart = Get-Date

        # Run build with logging
        $buildLogPath = Join-Path $LogsDir "build_log_$Timestamp.txt"
        & zig build > $buildLogPath 2>&1

        if ($LASTEXITCODE -eq 0) {
            $buildEnd = Get-Date
            $buildTime = ($buildEnd - $buildStart).TotalSeconds

            $binarySize = if (Test-Path "zig-out\bin\abi.exe") {
                $size = (Get-Item "zig-out\bin\abi.exe").Length
                "$([math]::Round($size / 1MB, 2)) MB"
            } else { "Binary size information not available" }

            $buildMetrics = "Build Performance Metrics - $(Get-Date)`n" +
                           "====================================`n`n" +
                           "Build Time: $([math]::Round($buildTime, 3))s`n" +
                           "Build Status: SUCCESS`n" +
                           "Build Log: $buildLogPath`n`n" +
                           "Binary Size Information:`n" +
                           "$binarySize"

            $buildMetrics | Out-File -FilePath (Join-Path $MetricsDir "build_metrics_$Timestamp.txt")
            Write-Success "Build completed successfully in $([math]::Round($buildTime, 3)) seconds"
        } else {
            Write-Warning "Build failed - check logs for details"
            $buildMetrics = "Build Performance Metrics - $(Get-Date)`n" +
                           "====================================`n`n" +
                           "Build Status: FAILED`n" +
                           "Build Log: $buildLogPath"

            $buildMetrics | Out-File -FilePath (Join-Path $MetricsDir "build_metrics_$Timestamp.txt")
        }
    }
    finally {
        Pop-Location
    }
}

# Runtime performance monitoring
function Monitor-RuntimePerformance {
    Write-Status "Monitoring runtime performance..."

    Push-Location $ProjectRoot

    try {
        if (Test-Path "zig-out\bin\abi.exe") {
            Write-Status "Testing application startup time..."

            $runtimeLogPath = Join-Path $LogsDir "runtime_test_$Timestamp.txt"
            $startTime = Get-Date

            # Start process and wait briefly
            $process = Start-Process -FilePath "zig-out\bin\abi.exe" -RedirectStandardOutput $runtimeLogPath -RedirectStandardError $runtimeLogPath -PassThru
            Start-Sleep -Seconds 3

            if (!$process.HasExited) {
                $process.Kill()
                $endTime = Get-Date
                $startupTime = ($endTime - $startTime).TotalSeconds

                $processInfo = Get-Process -Name abi -ErrorAction SilentlyContinue | Format-Table -AutoSize | Out-String
                $memoryUsage = Get-Process -Name abi -ErrorAction SilentlyContinue | Select-Object -ExpandProperty WorkingSet64 | ForEach-Object { "$([math]::Round($_ / 1MB, 2)) MB" }

                $runtimeMetrics = "Runtime Performance Metrics - $(Get-Date)`n" +
                                 "======================================`n`n" +
                                 "Application Status: RUNNING`n" +
                                 "Startup Time: $([math]::Round($startupTime, 3))s`n" +
                                 "Runtime Log: $runtimeLogPath`n`n" +
                                 "Process Information:`n" +
                                 "$processInfo`n" +
                                 "Memory Usage:`n" +
                                 "$memoryUsage"

                $runtimeMetrics | Out-File -FilePath (Join-Path $MetricsDir "runtime_metrics_$Timestamp.txt")
                Write-Success "Application started successfully in $([math]::Round($startupTime, 3)) seconds"
            } else {
                Write-Warning "Application failed to start properly"
                $runtimeMetrics = "Runtime Performance Metrics - $(Get-Date)`n" +
                                 "======================================`n`n" +
                                 "Application Status: FAILED`n" +
                                 "Runtime Log: $runtimeLogPath"

                $runtimeMetrics | Out-File -FilePath (Join-Path $MetricsDir "runtime_metrics_$Timestamp.txt")
            }
        } else {
            Write-Warning "Application binary not found"
        }
    }
    finally {
        Pop-Location
    }
}

# Code quality metrics
function Analyze-CodeQuality {
    Write-Status "Analyzing code quality..."

    Push-Location $ProjectRoot

    try {
        $zigFiles = Get-ChildItem -Path "src" -Filter "*.zig" -Recurse -File
        $totalLines = 0
        $totalFiles = $zigFiles.Count

        foreach ($file in $zigFiles) {
            $content = Get-Content $file.FullName -Raw
            $lineCount = ($content -split "`n").Count
            $totalLines += $lineCount
        }

        $todos = ($zigFiles | Get-Content | Select-String -Pattern "TODO|FIXME|XXX" | Measure-Object).Count
        $comments = ($zigFiles | Get-Content | Select-String -Pattern "//" | Measure-Object).Count

        $avgLinesPerFile = if ($totalFiles -gt 0) { [math]::Round($totalLines / $totalFiles, 1) } else { "N/A" }
        $codeQualityScore = if ($totalLines -gt 0) { "$([math]::Round(($comments * 100) / $totalLines, 1))% commented" } else { "N/A" }

        $codeQuality = "Code Quality Metrics - $(Get-Date)`n" +
                        "==============================`n`n" +
                        "Total Zig Files: $totalFiles`n" +
                        "Total Lines of Code: $totalLines`n" +
                        "Average Lines per File: $avgLinesPerFile`n" +
                        "Total Comments: $comments`n" +
                        "Total TODO/FIXME Items: $todos`n`n" +
                        "Code Quality Score: $codeQualityScore"

        $codeQuality | Out-File -FilePath (Join-Path $MetricsDir "code_quality_$Timestamp.txt")
        Write-Metric "Code quality analysis completed"
    }
    finally {
        Pop-Location
    }
}

# Generate comprehensive report
function Generate-Report {
    Write-Status "Generating comprehensive performance report..."

    $systemInfo = Get-Content (Join-Path $MetricsDir "system_info_$Timestamp.txt") -ErrorAction SilentlyContinue -Raw
    $buildMetrics = Get-Content (Join-Path $MetricsDir "build_metrics_$Timestamp.txt") -ErrorAction SilentlyContinue -Raw
    $runtimeMetrics = Get-Content (Join-Path $MetricsDir "runtime_metrics_$Timestamp.txt") -ErrorAction SilentlyContinue -Raw
    $codeQuality = Get-Content (Join-Path $MetricsDir "code_quality_$Timestamp.txt") -ErrorAction SilentlyContinue -Raw

    $buildStatus = if (Test-Path "zig-out\bin\abi.exe") { "Success" } else { "Failed" }
    $runtimeStatus = if ($runtimeMetrics -and $runtimeMetrics -match "RUNNING") { "Running" } else { "Issues" }

    $report = @"
# 📊 Performance Monitoring Report

Generated: $(Get-Date)
Timestamp: $Timestamp
Project: ABI Framework

## Executive Summary

This report provides a comprehensive overview of the ABI Framework's performance metrics, code quality, and system health.

## 📈 System Information

``````
$($systemInfo ? $systemInfo : "System information not available")
``````

## 🏗️ Build Performance

``````
$($buildMetrics ? $buildMetrics : "Build metrics not available")
``````

## ⚡ Runtime Performance

``````
$($runtimeMetrics ? $runtimeMetrics : "Runtime metrics not available")
``````

## 📝 Code Quality Metrics

``````
$($codeQuality ? $codeQuality : "Code quality metrics not available")
``````

## 📋 Recommendations

### Immediate Actions
$(if (!(Test-Path "zig-out\bin\abi.exe")) {
    "- [ ] Build the application: `zig build`"
})

### Performance Optimizations
- [ ] Review build times and optimize compilation flags
- [ ] Monitor memory usage patterns
- [ ] Analyze code complexity and refactoring opportunities
- [ ] Consider implementing performance regression tests

### Monitoring Improvements
- [ ] Set up automated alerts for performance degradation
- [ ] Implement comprehensive error tracking
- [ ] Add detailed profiling capabilities
- [ ] Establish performance baselines for key operations

## 📊 Key Metrics Summary

| Metric | Value | Status |
|--------|-------|--------|
| Build Status | $buildStatus | $(if (Test-Path "zig-out\bin\abi.exe") { "Good" } else { "Needs Attention" }) |
| Runtime Status | $runtimeStatus | $(if ($runtimeMetrics -and $runtimeMetrics -match "RUNNING") { "Good" } else { "Needs Attention" }) |

---

*This report was generated automatically by the ABI Framework Performance Monitoring System.*
"@

    $report | Out-File -FilePath $ReportFile -Encoding UTF8
    Write-Success "Comprehensive report generated: $ReportFile"
}

# Main monitoring function
function Invoke-Monitoring {
    Write-Status "Starting automated performance monitoring..."

    # Run all monitoring tasks
    Collect-SystemInfo
    Monitor-BuildPerformance
    Monitor-RuntimePerformance
    Analyze-CodeQuality
    Generate-Report

    Write-Success "Performance monitoring completed successfully!"
    Write-Metric "Report available at: $ReportFile"

    # Display summary
    Write-Host ""
    Write-Host "📋 Quick Summary:" -ForegroundColor Yellow
    Write-Host "=================="

    if (Test-Path "zig-out\bin\abi.exe") {
        Write-Host "✅ Application builds successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Application build failed" -ForegroundColor Red
    }

    $runtimeMetricsPath = Join-Path $MetricsDir "runtime_metrics_$Timestamp.txt"
    if (Test-Path $runtimeMetricsPath) {
        $runtimeContent = Get-Content $runtimeMetricsPath -Raw
        if ($runtimeContent -match "RUNNING") {
            Write-Host "✅ Application starts successfully" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Application startup issues detected" -ForegroundColor Yellow
        }
    }

    Write-Host "📊 Metrics collected in: $MetricsDir" -ForegroundColor Cyan
    Write-Host "📝 Logs available in: $LogsDir" -ForegroundColor Cyan
    Write-Host "📄 Full report: $ReportFile" -ForegroundColor Cyan
}

# Continuous monitoring loop
if ($Continuous) {
    Write-Status "Starting continuous monitoring (interval: $IntervalMinutes minutes)..."
    Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow

    while ($true) {
        Invoke-Monitoring
        Write-Status "Waiting $IntervalMinutes minutes until next monitoring cycle..."
        Start-Sleep -Seconds ($IntervalMinutes * 60)
    }
} else {
    # Run monitoring once
    Invoke-Monitoring
}
