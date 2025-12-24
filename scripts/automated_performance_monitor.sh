#!/bin/bash

# Automated Performance Monitoring System
# Continuously monitors performance metrics and generates reports

set -e

echo "📊 Automated Performance Monitoring System"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/reports"
METRICS_DIR="$REPORTS_DIR/metrics"
LOGS_DIR="$REPORTS_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$REPORTS_DIR/performance_report_$TIMESTAMP.md"

# Create directories
mkdir -p "$REPORTS_DIR"
mkdir -p "$METRICS_DIR"
mkdir -p "$LOGS_DIR"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_metric() {
    echo -e "${CYAN}[METRIC]${NC} $1"
}

# System information collection
collect_system_info() {
    print_status "Collecting system information..."

    cat > "$METRICS_DIR/system_info_$TIMESTAMP.txt" << EOF
System Information - $(date)
================================

CPU Information:
$(lscpu 2>/dev/null || echo "lscpu not available")

Memory Information:
$(free -h 2>/dev/null || echo "free not available")

Disk Information:
$(df -h 2>/dev/null || echo "df not available")

OS Information:
$(uname -a)

Uptime:
$(uptime)

Load Average:
$(cat /proc/loadavg 2>/dev/null || echo "Load average not available")
EOF
}

# Build performance monitoring
monitor_build_performance() {
    print_status "Monitoring build performance..."

    cd "$PROJECT_ROOT"

    # Time the build process
    local build_start=$(date +%s.%3N)
    if zig build > "$LOGS_DIR/build_log_$TIMESTAMP.txt" 2>&1; then
        local build_end=$(date +%s.%3N)
        local build_time=$(echo "$build_end - $build_start" | bc 2>/dev/null || echo "0")

        cat > "$METRICS_DIR/build_metrics_$TIMESTAMP.txt" << EOF
Build Performance Metrics - $(date)
====================================

Build Time: ${build_time}s
Build Status: SUCCESS
Build Log: $LOGS_DIR/build_log_$TIMESTAMP.txt

Binary Size Information:
$(ls -lh zig-out/bin/ 2>/dev/null || echo "Binary size information not available")
EOF
    else
        print_warning "Build failed - check logs for details"
        cat > "$METRICS_DIR/build_metrics_$TIMESTAMP.txt" << EOF
Build Performance Metrics - $(date)
====================================

Build Status: FAILED
Build Log: $LOGS_DIR/build_log_$TIMESTAMP.txt
EOF
    fi
}

# Runtime performance monitoring
monitor_runtime_performance() {
    print_status "Monitoring runtime performance..."

    cd "$PROJECT_ROOT"

    # Test basic functionality
    if [ -f "zig-out/bin/abi" ]; then
        print_status "Testing application startup time..."

        local startup_start=$(date +%s.%3N)
        timeout 10s ./zig-out/bin/abi > "$LOGS_DIR/runtime_test_$TIMESTAMP.txt" 2>&1 &
        local pid=$!
        sleep 2

        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
            local startup_end=$(date +%s.%3N)
            local startup_time=$(echo "$startup_end - $startup_start" | bc 2>/dev/null || echo "0")

            cat > "$METRICS_DIR/runtime_metrics_$TIMESTAMP.txt" << EOF
Runtime Performance Metrics - $(date)
======================================

Application Status: RUNNING
Startup Time: ${startup_time}s
Runtime Log: $LOGS_DIR/runtime_test_$TIMESTAMP.txt

Process Information:
$(ps aux | grep abi | grep -v grep || echo "Process information not available")

Memory Usage:
$(ps aux | grep abi | grep -v grep | awk '{print $6}' || echo "Memory usage not available")
EOF
        else
            print_warning "Application failed to start properly"
            cat > "$METRICS_DIR/runtime_metrics_$TIMESTAMP.txt" << EOF
Runtime Performance Metrics - $(date)
======================================

Application Status: FAILED
Runtime Log: $LOGS_DIR/runtime_test_$TIMESTAMP.txt
EOF
        fi
    else
        print_warning "Application binary not found"
    fi
}

# Code quality metrics
analyze_code_quality() {
    print_status "Analyzing code quality..."

    cd "$PROJECT_ROOT"

    # Count lines of code
    local zig_files=$(find src -name "*.zig" -type f)
    local total_lines=0
    local total_files=0

    for file in $zig_files; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file")
            total_lines=$((total_lines + lines))
            total_files=$((total_files + 1))
        fi
    done

    # Check for task markers
    local todos=$(grep -r "TASK_MARKER" src/ --include="*.zig" 2>/dev/null | wc -l)
    local comments=$(grep -r "//" src/ --include="*.zig" 2>/dev/null | wc -l)

    cat > "$METRICS_DIR/code_quality_$TIMESTAMP.txt" << EOF
Code Quality Metrics - $(date)
==============================

Total Zig Files: $total_files
Total Lines of Code: $total_lines
Average Lines per File: $((total_lines / total_files))
Total Comments: $comments
Total Task Markers: $todos

Code Quality Score: $(( (comments * 100) / total_lines ))% commented
EOF
}

# Performance regression detection
detect_performance_regression() {
    print_status "Detecting performance regressions..."

    cd "$PROJECT_ROOT"

    # Compare with previous metrics if available
    local prev_metrics=$(ls -t "$METRICS_DIR"/*_metrics_*.txt 2>/dev/null | head -2 | tail -1)

    if [ -n "$prev_metrics" ]; then
        print_status "Comparing with previous metrics: $prev_metrics"

        # Simple comparison - in production, you'd want more sophisticated analysis
        cat > "$METRICS_DIR/regression_analysis_$TIMESTAMP.txt" << EOF
Performance Regression Analysis - $(date)
========================================

Previous Metrics: $prev_metrics
Current Metrics: $METRICS_DIR/*_metrics_$TIMESTAMP.txt

Automation: Regression detection requires historical data analysis.
Consider implementing statistical analysis for more accurate regression detection.
EOF
    else
        print_status "No previous metrics found - establishing baseline"
        cat > "$METRICS_DIR/regression_analysis_$TIMESTAMP.txt" << EOF
Performance Regression Analysis - $(date)
========================================

Status: Establishing baseline metrics
No previous data available for comparison.
EOF
    fi
}

# Generate comprehensive report
generate_report() {
    print_status "Generating comprehensive performance report..."

    cat > "$REPORT_FILE" << EOF
# 📊 Performance Monitoring Report

Generated: $(date)
Timestamp: $TIMESTAMP
Project: ABI Framework

## Executive Summary

This report provides a comprehensive overview of the ABI Framework's performance metrics, code quality, and system health.

## 📈 System Information

\`\`\`
$(cat "$METRICS_DIR/system_info_$TIMESTAMP.txt" 2>/dev/null || echo "System information not available")
\`\`\`

## 🏗️ Build Performance

\`\`\`
$(cat "$METRICS_DIR/build_metrics_$TIMESTAMP.txt" 2>/dev/null || echo "Build metrics not available")
\`\`\`

## ⚡ Runtime Performance

\`\`\`
$(cat "$METRICS_DIR/runtime_metrics_$TIMESTAMP.txt" 2>/dev/null || echo "Runtime metrics not available")
\`\`\`

## 📝 Code Quality Metrics

\`\`\`
$(cat "$METRICS_DIR/code_quality_$TIMESTAMP.txt" 2>/dev/null || echo "Code quality metrics not available")
\`\`\`

## 🔍 Performance Regression Analysis

\`\`\`
$(cat "$METRICS_DIR/regression_analysis_$TIMESTAMP.txt" 2>/dev/null || echo "Regression analysis not available")
\`\`\`

## 📋 Recommendations

### Immediate Actions
$(if [ ! -f "zig-out/bin/abi" ]; then
    echo "- [ ] Build the application: \`zig build\`"
fi)

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
| Build Status | $(if [ -f "zig-out/bin/abi" ]; then echo "✅ Success"; else echo "❌ Failed"; fi) | $(if [ -f "zig-out/bin/abi" ]; then echo "Good"; else echo "Needs Attention"; fi) |
| Code Quality | $(cat "$METRICS_DIR/code_quality_$TIMESTAMP.txt" 2>/dev/null | grep "Code Quality Score" | cut -d: -f2 || echo "Unknown") | Good |
| Runtime Status | $(cat "$METRICS_DIR/runtime_metrics_$TIMESTAMP.txt" 2>/dev/null | grep "Application Status" | cut -d: -f2 || echo "Unknown") | $(cat "$METRICS_DIR/runtime_metrics_$TIMESTAMP.txt" 2>/dev/null | grep "Application Status" | grep -q "RUNNING" && echo "Good" || echo "Needs Attention") |

---

*This report was generated automatically by the ABI Framework Performance Monitoring System.*
EOF

    print_success "Comprehensive report generated: $REPORT_FILE"
}

# Main monitoring function
main() {
    print_status "Starting automated performance monitoring..."

    # Run all monitoring tasks
    collect_system_info
    monitor_build_performance
    monitor_runtime_performance
    analyze_code_quality
    detect_performance_regression
    generate_report

    print_success "Performance monitoring completed successfully!"
    print_metric "Report available at: $REPORT_FILE"

    # Display summary
    echo
    echo "📋 Quick Summary:"
    echo "=================="

    if [ -f "zig-out/bin/abi" ]; then
        echo "✅ Application builds successfully"
    else
        echo "❌ Application build failed"
    fi

    if [ -f "$METRICS_DIR/runtime_metrics_$TIMESTAMP.txt" ]; then
        if grep -q "RUNNING" "$METRICS_DIR/runtime_metrics_$TIMESTAMP.txt" 2>/dev/null; then
            echo "✅ Application starts successfully"
        else
            echo "⚠️  Application startup issues detected"
        fi
    fi

    echo "📊 Metrics collected in: $METRICS_DIR"
    echo "📝 Logs available in: $LOGS_DIR"
    echo "📄 Full report: $REPORT_FILE"
}

# Run the monitoring system
main "$@"

