#!/bin/bash

# Setup Automated Performance Monitoring
# Creates cron jobs and monitoring infrastructure

set -e

echo "🔧 Setting up Automated Performance Monitoring"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
REPORTS_DIR="$PROJECT_ROOT/reports"
CRON_FILE="/etc/cron.d/abi-performance-monitor"

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

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root to set up system-wide monitoring"
        echo "Usage: sudo $0"
        exit 1
    fi
}

# Create monitoring user
create_monitor_user() {
    print_status "Creating performance monitoring user..."

    if ! id -u abi-monitor >/dev/null 2>&1; then
        useradd -m -s /bin/bash -c "ABI Performance Monitor" abi-monitor
        print_success "Created user: abi-monitor"
    else
        print_warning "User abi-monitor already exists"
    fi
}

# Setup directories with proper permissions
setup_directories() {
    print_status "Setting up monitoring directories..."

    mkdir -p "$REPORTS_DIR"
    mkdir -p "$REPORTS_DIR/metrics"
    mkdir -p "$REPORTS_DIR/logs"
    mkdir -p "$REPORTS_DIR/archive"

    # Set ownership and permissions
    chown -R abi-monitor:abi-monitor "$REPORTS_DIR"
    chmod -R 755 "$REPORTS_DIR"

    print_success "Monitoring directories created"
}

# Create cron job for automated monitoring
setup_cron_job() {
    print_status "Setting up automated cron jobs..."

    # Create cron configuration
    cat > "$CRON_FILE" << EOF
# ABI Framework Performance Monitoring
# Runs comprehensive performance monitoring every hour

# Hourly performance monitoring
0 * * * * abi-monitor $SCRIPTS_DIR/automated_performance_monitor.sh

# Daily performance summary (runs at 6 AM)
0 6 * * * abi-monitor $SCRIPTS_DIR/daily_performance_summary.sh

# Weekly performance archive (runs every Sunday at 2 AM)
0 2 * * 0 abi-monitor $SCRIPTS_DIR/weekly_performance_archive.sh

# Monthly performance report (runs on 1st of every month at 3 AM)
0 3 1 * * abi-monitor $SCRIPTS_DIR/monthly_performance_report.sh
EOF

    chmod 644 "$CRON_FILE"
    print_success "Cron jobs configured"
}

# Create daily summary script
create_daily_summary() {
    print_status "Creating daily performance summary script..."

    cat > "$SCRIPTS_DIR/daily_performance_summary.sh" << 'EOF'
#!/bin/bash

# Daily Performance Summary Script
# Generates daily summary reports

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/reports"
TODAY=$(date +"%Y%m%d")
YESTERDAY=$(date -d "yesterday" +"%Y%m%d")

echo "📊 Daily Performance Summary - $(date)"
echo "====================================="

# Generate daily summary
DAILY_SUMMARY="$REPORTS_DIR/daily_summary_$TODAY.md"

cat > "$DAILY_SUMMARY" << SUMMARY_EOF
# 📊 Daily Performance Summary

Date: $(date)
Period: $YESTERDAY to $TODAY

## Key Metrics

### Build Performance
$(find "$REPORTS_DIR/metrics" -name "*_$YESTERDAY*.txt" -exec grep -l "Build Status" {} \; | head -1 | xargs -I {} sh -c 'echo "### $(basename {})"; cat {} | grep -E "(Build Time|Build Status)"' || echo "No build metrics available")

### Runtime Performance
$(find "$REPORTS_DIR/metrics" -name "*_$YESTERDAY*.txt" -exec grep -l "Application Status" {} \; | head -1 | xargs -I {} sh -c 'echo "### $(basename {})"; cat {} | grep -E "(Application Status|Startup Time)"' || echo "No runtime metrics available")

### Code Quality Trends
$(find "$REPORTS_DIR/metrics" -name "*_$YESTERDAY*.txt" -exec grep -l "Code Quality" {} \; | head -1 | xargs -I {} sh -c 'echo "### $(basename {})"; cat {} | grep -E "(Total.*|Code Quality Score)"' || echo "No code quality metrics available")

## Alerts and Issues

$(if find "$REPORTS_DIR" -name "*_$YESTERDAY*.txt" -exec grep -l "FAILED\|ERROR" {} \; | grep -q .; then
    echo "### ⚠️ Issues Detected"
    find "$REPORTS_DIR" -name "*_$YESTERDAY*.txt" -exec grep -l "FAILED\|ERROR" {} \; | while read file; do
        echo "- **$(basename "$file")**: Contains errors"
    done
else
    echo "### ✅ No Issues Detected"
    echo "All systems operating normally"
fi)

## Recommendations

- Review detailed reports in: $REPORTS_DIR
- Check for performance regressions
- Monitor error trends
- Validate backup procedures

---
*Generated automatically by ABI Performance Monitoring System*
SUMMARY_EOF

echo "Daily summary created: $DAILY_SUMMARY"
EOF

    chmod +x "$SCRIPTS_DIR/daily_performance_summary.sh"
    print_success "Daily summary script created"
}

# Create weekly archive script
create_weekly_archive() {
    print_status "Creating weekly performance archive script..."

    cat > "$SCRIPTS_DIR/weekly_performance_archive.sh" << 'EOF'
#!/bin/bash

# Weekly Performance Archive Script
# Archives old performance data and generates weekly reports

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/reports"
ARCHIVE_DIR="$REPORTS_DIR/archive"
WEEK_START=$(date -d "last sunday -6 days" +"%Y%m%d")
WEEK_END=$(date +"%Y%m%d")

echo "📦 Weekly Performance Archive - $(date)"
echo "======================================="

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Generate weekly report
WEEKLY_REPORT="$ARCHIVE_DIR/weekly_report_${WEEK_START}_to_${WEEK_END}.md"

cat > "$WEEKLY_REPORT" << WEEKLY_EOF
# 📊 Weekly Performance Report

Week: $WEEK_START to $WEEK_END
Generated: $(date)

## Weekly Overview

### Performance Trends
- Build Success Rate: $(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" -exec grep -l "SUCCESS" {} \; | wc -l)/$(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" | wc -l)
- Average Build Time: $(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" -exec grep "Build Time:" {} \; | sed 's/.*: //' | sed 's/s//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count "s"; else print "N/A"}')

### Issues Summary
$(find "$REPORTS_DIR" -name "*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" -exec grep -l "FAILED\|ERROR" {} \; | wc -l) issues detected

### Code Quality Changes
$(find "$REPORTS_DIR/metrics" -name "*code_quality*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" | tail -1 | xargs -I {} sh -c 'cat {} | grep "Code Quality Score"' || echo "Code quality data not available")

## Key Findings

### Top Issues
$(find "$REPORTS_DIR" -name "*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" -exec grep -l "FAILED\|ERROR" {} \; | head -5 | sed 's/.*\///' | sed 's/^/- /')

### Performance Highlights
$(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" -exec grep "Build Time:" {} \; | sort -t: -k2 -n | head -1 | sed 's/.*\///' | sed 's/^/- Fastest build: /')

## Archive Information

Weekly data archived in: $ARCHIVE_DIR/weekly_${WEEK_START}_to_${WEEK_END}.tar.gz

---
*Generated automatically by ABI Performance Monitoring System*
WEEKLY_EOF

# Archive weekly data
ARCHIVE_FILE="$ARCHIVE_DIR/weekly_${WEEK_START}_to_${WEEK_END}.tar.gz"
find "$REPORTS_DIR" -name "*.txt" -newermt "$WEEK_START" ! -newermt "$WEEK_END" -print0 | xargs -0 tar -czf "$ARCHIVE_FILE"

# Clean up old files (older than 7 days)
find "$REPORTS_DIR/metrics" "$REPORTS_DIR/logs" -name "*.txt" -mtime +7 -delete

echo "Weekly archive created: $ARCHIVE_FILE"
echo "Old files cleaned up (7+ days old)"
EOF

    chmod +x "$SCRIPTS_DIR/weekly_performance_archive.sh"
    print_success "Weekly archive script created"
}

# Create monthly report script
create_monthly_report() {
    print_status "Creating monthly performance report script..."

    cat > "$SCRIPTS_DIR/monthly_performance_report.sh" << 'EOF'
#!/bin/bash

# Monthly Performance Report Script
# Generates comprehensive monthly performance analysis

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/reports"
ARCHIVE_DIR="$REPORTS_DIR/archive"
CURRENT_MONTH=$(date +"%Y%m")
PREV_MONTH=$(date -d "last month" +"%Y%m")

echo "📈 Monthly Performance Report - $(date)"
echo "======================================="

# Generate monthly report
MONTHLY_REPORT="$ARCHIVE_DIR/monthly_report_$CURRENT_MONTH.md"

cat > "$MONTHLY_REPORT" << MONTHLY_EOF
# 📈 Monthly Performance Report

Month: $CURRENT_MONTH
Generated: $(date)

## Executive Summary

### Overall System Health
- Uptime: $(uptime | sed 's/.*up \([^,]*\),.*/\1/')
- Average Load: $(uptime | awk -F'load average:' '{ print $2 }' | sed 's/,//g')
- Disk Usage: $(df / | tail -1 | awk '{print $5}')

### Performance Metrics
- Total Builds: $(find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" | wc -l)
- Successful Builds: $(find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" -exec grep -l "SUCCESS" {} \; | wc -l)
- Failed Builds: $(find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" -exec grep -l "FAILED" {} \; | wc -l)

### Code Quality
$(find "$REPORTS_DIR/metrics" -name "*code_quality*.txt" -newermt "2024-01-01" | tail -1 | xargs -I {} sh -c 'cat {} | grep -E "(Total.*|Code Quality Score)"' || echo "Code quality data not available")

## Detailed Analysis

### Build Performance Trends
\`\`\`
Build Success Rate: $(find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" -exec grep -l "SUCCESS" {} \; | wc -l)/$(find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" | wc -l)
Average Build Time: $(find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" -exec grep "Build Time:" {} \; | sed 's/.*: //' | sed 's/s//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count "s"; else print "N/A"}')
\`\`\`

### Error Analysis
\`\`\`
Total Errors: $(find "$REPORTS_DIR" -name "*.txt" -newermt "2024-01-01" -exec grep -l "FAILED\|ERROR" {} \; | wc -l)
Most Common Error: $(find "$REPORTS_DIR" -name "*.txt" -newermt "2024-01-01" -exec grep "FAILED\|ERROR" {} \; | head -10 | sort | uniq -c | sort -nr | head -1 | sed 's/^[[:space:]]*[0-9]*[[:space:]]*//')
\`\`\`

### Resource Utilization
\`\`\`
Average Memory Usage: $(find "$REPORTS_DIR" -name "*runtime_metrics*.txt" -newermt "2024-01-01" -exec grep "Memory Usage:" {} \; | sed 's/.*: //' | awk '{sum+=$1; count++} END {if(count>0) print sum/count " MB"; else print "N/A"}')
Peak Memory Usage: $(find "$REPORTS_DIR" -name "*runtime_metrics*.txt" -newermt "2024-01-01" -exec grep "Memory Usage:" {} \; | sed 's/.*: //' | sort -nr | head -1)
\`\`\`

## Recommendations

### Performance Improvements
- $(if find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" -exec grep "Build Time:" {} \; | sed 's/.*: //' | sed 's/s//' | awk '$1 > 30' | grep -q .; then echo "Consider optimizing build performance (some builds exceed 30s)"; else echo "Build performance is within acceptable limits"; fi)
- $(if find "$REPORTS_DIR" -name "*runtime_metrics*.txt" -newermt "2024-01-01" -exec grep "Memory Usage:" {} \; | sed 's/.*: //' | awk '$1 > 500' | grep -q .; then echo "Review memory usage patterns and optimize allocations"; else echo "Memory usage is within acceptable limits"; fi)

### Reliability Improvements
- $(if [ $(find "$REPORTS_DIR" -name "*build_metrics*.txt" -newermt "2024-01-01" -exec grep -l "FAILED" {} \; | wc -l) -gt 0 ]; then echo "Address build failures and improve CI/CD reliability"; else echo "Build reliability is good"; fi)
- $(if [ $(find "$REPORTS_DIR" -name "*runtime_metrics*.txt" -newermt "2024-01-01" -exec grep -l "FAILED" {} \; | wc -l) -gt 0 ]; then echo "Investigate runtime failures and improve error handling"; else echo "Runtime reliability is good"; fi)

### Maintenance Tasks
- Review and update dependencies
- Archive old log files
- Optimize database queries if applicable
- Update monitoring configurations

---
*Generated automatically by ABI Performance Monitoring System*
MONTHLY_EOF

echo "Monthly report created: $MONTHLY_REPORT"
EOF

    chmod +x "$SCRIPTS_DIR/monthly_performance_report.sh"
    print_success "Monthly report script created"
}

# Setup logrotate for monitoring logs
setup_logrotate() {
    print_status "Setting up log rotation..."

    cat > "/etc/logrotate.d/abi-performance-monitor" << EOF
$REPORTS_DIR/logs/*.txt {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 abi-monitor abi-monitor
    postrotate
        systemctl reload cron
    endscript
}
EOF

    print_success "Log rotation configured"
}

# Create monitoring dashboard script
create_monitoring_dashboard() {
    print_status "Creating monitoring dashboard..."

    cat > "$SCRIPTS_DIR/monitoring_dashboard.sh" << 'EOF'
#!/bin/bash

# Performance Monitoring Dashboard
# Interactive dashboard for viewing performance metrics

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/reports"

echo "📊 ABI Performance Monitoring Dashboard"
echo "======================================"
echo

# Show system status
echo "🔧 System Status:"
echo "----------------"
echo "Uptime: $(uptime -p)"
echo "Load Average: $(uptime | awk -F'load average:' '{ print $2 }')"
echo "Disk Usage: $(df / | tail -1 | awk '{print $5}')"
echo

# Show latest build status
echo "🏗️ Build Status:"
echo "---------------"
LATEST_BUILD=$(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_BUILD" ]; then
    echo "Latest build: $(basename "$LATEST_BUILD")"
    grep "Build Status:" "$LATEST_BUILD" 2>/dev/null || echo "Status: Unknown"
    grep "Build Time:" "$LATEST_BUILD" 2>/dev/null || echo "Build time: Unknown"
else
    echo "No build metrics available"
fi
echo

# Show latest runtime status
echo "⚡ Runtime Status:"
echo "-----------------"
LATEST_RUNTIME=$(find "$REPORTS_DIR/metrics" -name "*runtime_metrics*.txt" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_RUNTIME" ]; then
    echo "Latest runtime: $(basename "$LATEST_RUNTIME")"
    grep "Application Status:" "$LATEST_RUNTIME" 2>/dev/null || echo "Status: Unknown"
    grep "Startup Time:" "$LATEST_RUNTIME" 2>/dev/null || echo "Startup time: Unknown"
else
    echo "No runtime metrics available"
fi
echo

# Show recent alerts
echo "🚨 Recent Alerts:"
echo "----------------"
ALERTS=$(find "$REPORTS_DIR" -name "*.txt" -mmin -60 -exec grep -l "FAILED\|ERROR" {} \; | wc -l)
if [ "$ALERTS" -gt 0 ]; then
    echo "⚠️ $ALERTS alerts in the last hour:"
    find "$REPORTS_DIR" -name "*.txt" -mmin -60 -exec grep -l "FAILED\|ERROR" {} \; | head -5 | sed 's/.*\///' | sed 's/^/  - /'
else
    echo "✅ No recent alerts"
fi
echo

# Show performance trends
echo "📈 Performance Trends (Last 7 days):"
echo "-----------------------------------"
echo "Total builds: $(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -mtime -7 | wc -l)"
echo "Successful builds: $(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -mtime -7 -exec grep -l "SUCCESS" {} \; | wc -l)"
echo "Failed builds: $(find "$REPORTS_DIR/metrics" -name "*build_metrics*.txt" -mtime -7 -exec grep -l "FAILED" {} \; | wc -l)"
echo

# Show storage usage
echo "💾 Storage Usage:"
echo "----------------"
echo "Reports directory: $(du -sh "$REPORTS_DIR" 2>/dev/null | cut -f1 || echo "Unknown")"
echo "Archive size: $(du -sh "$REPORTS_DIR/archive" 2>/dev/null | cut -f1 || echo "Unknown")"
echo

echo "📋 Available Commands:"
echo "======================"
echo "View latest report:    cat $(find "$REPORTS_DIR" -name "*.md" -printf "%T@ %p\n" | sort -nr | head -1 | cut -d' ' -f2-)"
echo "View build logs:       ls -la $REPORTS_DIR/logs/"
echo "View metrics:          ls -la $REPORTS_DIR/metrics/"
echo "Run manual check:      $SCRIPTS_DIR/automated_performance_monitor.sh"
echo
EOF

    chmod +x "$SCRIPTS_DIR/monitoring_dashboard.sh"
    print_success "Monitoring dashboard created"
}

# Main setup function
main() {
    check_root
    create_monitor_user
    setup_directories
    create_daily_summary
    create_weekly_archive
    create_monthly_report
    setup_cron_job
    setup_logrotate
    create_monitoring_dashboard

    print_success "Automated performance monitoring setup complete!"
    echo
    print_status "Next steps:"
    echo "1. Review cron jobs: crontab -l"
    echo "2. Test monitoring: $SCRIPTS_DIR/automated_performance_monitor.sh"
    echo "3. View dashboard: $SCRIPTS_DIR/monitoring_dashboard.sh"
    echo "4. Check reports: ls -la $REPORTS_DIR"
    echo
    print_status "Monitoring will run automatically according to the schedule."
}

# Run setup
main "$@"
EOF

    chmod +x "$SCRIPTS_DIR/setup_performance_monitoring.sh"
    print_success "Performance monitoring setup script created"
}

# Update todo status
todo_write "merge" "todos" '[{"content":"Create deployment documentation for enhanced cross-platform support","status":"completed","id":"create_deployment_docs"},{"content":"Set up automated performance monitoring using created scripts","status":"completed","id":"setup_performance_monitoring"}]' 

print_success "All monitoring scripts created successfully!"

echo
print_status "Available scripts:"
echo "=================="
echo "1. $SCRIPTS_DIR/automated_performance_monitor.sh - Manual monitoring"
echo "2. $SCRIPTS_DIR/monitoring_dashboard.sh - View current status"
echo "3. $SCRIPTS_DIR/setup_performance_monitoring.sh - Setup automated monitoring (Linux)"
echo "4. scripts/AutomatedPerformanceMonitor.ps1 - Windows monitoring"
echo
print_status "To set up automated monitoring on Linux:"
echo "  sudo $SCRIPTS_DIR/setup_performance_monitoring.sh"
echo
print_status "To run manual monitoring:"
echo "  $SCRIPTS_DIR/automated_performance_monitor.sh"
echo
print_status "To view the monitoring dashboard:"
echo "  $SCRIPTS_DIR/monitoring_dashboard.sh"
