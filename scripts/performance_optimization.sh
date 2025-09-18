#!/bin/bash

# Performance Optimization Analysis Script
# Analyzes current performance and suggests optimizations

set -e

echo "⚡ Performance Optimization Analysis"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="$PROJECT_ROOT/benchmarks"
SRC_DIR="$PROJECT_ROOT/src"
BUILD_DIR="$PROJECT_ROOT/zig-out"

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

# Run comprehensive benchmarks
run_benchmarks() {
    print_status "Running comprehensive benchmark suite..."

    cd "$PROJECT_ROOT"

    if ! zig build benchmark 2>/dev/null; then
        print_warning "Benchmark build failed, trying alternative..."

        # Run individual benchmarks
        if [ -f "$BENCHMARK_DIR/main.zig" ]; then
            zig run "$BENCHMARK_DIR/main.zig" -- all --export --format=json --output=perf_analysis_results
            print_success "Benchmark analysis completed"
        else
            print_warning "No benchmark suite found"
        fi
    else
        print_success "Benchmark suite executed successfully"
    fi
}

# Analyze build optimizations
analyze_build_optimizations() {
    print_status "Analyzing build optimization options..."

    cd "$PROJECT_ROOT"

    # Test different optimization levels
    OPTIMIZATION_LEVELS=("Debug" "ReleaseSafe" "ReleaseFast" "ReleaseSmall")

    echo "## Build Optimization Analysis" > /tmp/build_analysis.md
    echo "" >> /tmp/build_analysis.md
    echo "| Optimization Level | Build Time | Binary Size | Performance |" >> /tmp/build_analysis.md
    echo "|-------------------|------------|-------------|-------------|" >> /tmp/build_analysis.md

    for opt in "${OPTIMIZATION_LEVELS[@]}"; do
        print_status "Testing $opt optimization..."

        start_time=$(date +%s.%3N)

        if zig build -Doptimize="$opt" >/dev/null 2>&1; then
            end_time=$(date +%s.%3N)
            build_time=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "N/A")

            # Get binary size
            if [ -f "$BUILD_DIR/bin/abi" ]; then
                binary_size=$(du -h "$BUILD_DIR/bin/abi" | cut -f1)
            else
                binary_size="N/A"
            fi

            echo "| $opt | ${build_time}s | $binary_size | TBD |" >> /tmp/build_analysis.md
            print_success "$opt build completed in ${build_time}s"
        else
            echo "| $opt | Failed | N/A | N/A |" >> /tmp/build_analysis.md
            print_warning "$opt build failed"
        fi
    done
}

# Analyze SIMD usage
analyze_simd_usage() {
    print_status "Analyzing SIMD optimization opportunities..."

    cd "$PROJECT_ROOT"

    # Search for SIMD-related code
    SIMD_FILES=$(find "$SRC_DIR" -name "*.zig" -exec grep -l "simd\|SIMD\|@Vector\|std.simd" {} \;)

    echo "## SIMD Usage Analysis" > /tmp/simd_analysis.md
    echo "" >> /tmp/simd_analysis.md
    echo "### Files using SIMD:" >> /tmp/simd_analysis.md

    if [ -n "$SIMD_FILES" ]; then
        echo "$SIMD_FILES" | while read -r file; do
            echo "- \`$file\`" >> /tmp/simd_analysis.md
            # Count SIMD operations in each file
            simd_count=$(grep -c "simd\|SIMD\|@Vector\|std.simd" "$file" 2>/dev/null || echo "0")
            echo "  - SIMD operations: $simd_count" >> /tmp/simd_analysis.md
        done
        print_success "Found $(echo "$SIMD_FILES" | wc -l) files using SIMD"
    else
        echo "No SIMD usage detected" >> /tmp/simd_analysis.md
        print_warning "No SIMD usage detected in codebase"
    fi

    # Suggest SIMD optimizations
    echo "" >> /tmp/simd_analysis.md
    echo "### SIMD Optimization Recommendations:" >> /tmp/simd_analysis.md
    echo "- Consider using @Vector for data-parallel operations" >> /tmp/simd_analysis.md
    echo "- Leverage std.simd for cross-platform SIMD operations" >> /tmp/simd_analysis.md
    echo "- Profile SIMD vs scalar performance for critical paths" >> /tmp/simd_analysis.md
}

# Analyze memory usage patterns
analyze_memory_usage() {
    print_status "Analyzing memory usage patterns..."

    cd "$PROJECT_ROOT"

    # Search for memory allocation patterns
    ALLOC_PATTERNS=("allocator\.alloc" "allocator\.create" "ArrayList" "std\.heap")

    echo "## Memory Usage Analysis" > /tmp/memory_analysis.md
    echo "" >> /tmp/memory_analysis.md

    for pattern in "${ALLOC_PATTERNS[@]}"; do
        count=$(find "$SRC_DIR" -name "*.zig" -exec grep -l "$pattern" {} \; | wc -l)
        echo "- $pattern: $count files" >> /tmp/memory_analysis.md
    done

    # Check for memory leaks detection
    if grep -r "memory.*leak\|leak.*memory" "$SRC_DIR" >/dev/null 2>&1; then
        echo "- Memory leak detection: ENABLED" >> /tmp/memory_analysis.md
        print_success "Memory leak detection is enabled"
    else
        echo "- Memory leak detection: NOT FOUND" >> /tmp/memory_analysis.md
        print_warning "Consider enabling memory leak detection"
    fi
}

# Generate performance recommendations
generate_recommendations() {
    print_status "Generating performance optimization recommendations..."

    echo "## Performance Optimization Recommendations" > /tmp/perf_recommendations.md
    echo "" >> /tmp/perf_recommendations.md

    # Build optimization recommendations
    if [ -f /tmp/build_analysis.md ]; then
        echo "### Build Optimizations:" >> /tmp/perf_recommendations.md
        echo "- Use ReleaseFast for production deployments" >> /tmp/perf_recommendations.md
        echo "- Consider ReleaseSmall for memory-constrained environments" >> /tmp/perf_recommendations.md
        echo "- Enable LTO (-Dlto=true) for better optimization" >> /tmp/perf_recommendations.md
        echo "" >> /tmp/perf_recommendations.md
    fi

    # SIMD recommendations
    if [ -f /tmp/simd_analysis.md ]; then
        echo "### SIMD Optimizations:" >> /tmp/perf_recommendations.md
        echo "- Profile critical computational loops for SIMD conversion" >> /tmp/perf_recommendations.md
        echo "- Use target-specific SIMD levels (AVX2, AVX-512) where beneficial" >> /tmp/perf_recommendations.md
        echo "- Consider using @reduce for vector operations" >> /tmp/perf_recommendations.md
        echo "" >> /tmp/perf_recommendations.md
    fi

    # Memory optimization recommendations
    echo "### Memory Optimizations:" >> /tmp/perf_recommendations.md
    echo "- Use arena allocators for short-lived allocations" >> /tmp/perf_recommendations.md
    echo "- Consider object pooling for frequently allocated objects" >> /tmp/perf_recommendations.md
    echo "- Profile memory usage with heap analysis tools" >> /tmp/perf_recommendations.md
    echo "" >> /tmp/perf_recommendations.md

    # General recommendations
    echo "### General Performance Tips:" >> /tmp/perf_recommendations.md
    echo "- Enable Tracy profiler for detailed performance analysis" >> /tmp/perf_recommendations.md
    echo "- Use comptime where possible for compile-time computation" >> /tmp/perf_recommendations.md
    echo "- Profile before optimizing - measure performance impact" >> /tmp/perf_recommendations.md
    echo "- Consider caching for expensive computations" >> /tmp/perf_recommendations.md
}

# Create performance report
create_performance_report() {
    print_status "Creating comprehensive performance report..."

    REPORT_FILE="$PROJECT_ROOT/PERFORMANCE_OPTIMIZATION_REPORT.md"

    {
        echo "# Performance Optimization Analysis Report"
        echo ""
        echo "Generated on: $(date)"
        echo "Zig Version: $(zig version 2>/dev/null || echo 'Unknown')"
        echo ""

        if [ -f /tmp/build_analysis.md ]; then
            cat /tmp/build_analysis.md
            echo ""
        fi

        if [ -f /tmp/simd_analysis.md ]; then
            cat /tmp/simd_analysis.md
            echo ""
        fi

        if [ -f /tmp/memory_analysis.md ]; then
            cat /tmp/memory_analysis.md
            echo ""
        fi

        if [ -f /tmp/perf_recommendations.md ]; then
            cat /tmp/perf_recommendations.md
            echo ""
        fi

        echo "## Next Steps"
        echo ""
        echo "1. Review benchmark results and identify bottlenecks"
        echo "2. Implement recommended optimizations incrementally"
        echo "3. Re-run benchmarks to measure performance improvements"
        echo "4. Consider cross-platform performance testing"
        echo "5. Set up continuous performance monitoring"

    } > "$REPORT_FILE"

    print_success "Performance report saved to: $REPORT_FILE"
}

# Main execution
main() {
    run_benchmarks
    analyze_build_optimizations
    analyze_simd_usage
    analyze_memory_usage
    generate_recommendations
    create_performance_report

    print_success "Performance optimization analysis completed"
    print_status "Check PERFORMANCE_OPTIMIZATION_REPORT.md for detailed findings"
}

# Run main function
main "$@"
