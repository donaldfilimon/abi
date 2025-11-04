#!/bin/bash
# Build size measurement script
# Compares different optimization modes and feature configurations

set -e

echo "================================================"
echo "ABI Build Size Comparison"
echo "================================================"
echo ""

# Clean previous builds
rm -rf zig-out zig-cache 2>/dev/null || true

# Function to build and measure
measure_build() {
    local name="$1"
    local flags="$2"
    
    echo "Building: $name"
    echo "Flags: $flags"
    
    # Build
    eval "zig build $flags" > /dev/null 2>&1 || {
        echo "  ❌ Build failed"
        echo ""
        return 1
    }
    
    # Measure size
    if [ -f "zig-out/bin/abi" ]; then
        local size=$(stat -f%z "zig-out/bin/abi" 2>/dev/null || stat -c%s "zig-out/bin/abi" 2>/dev/null)
        local size_kb=$((size / 1024))
        local size_mb=$((size_kb / 1024))
        echo "  ✅ Size: ${size_kb} KB (${size_mb} MB)"
        echo ""
        return 0
    else
        echo "  ⚠️  Binary not found"
        echo ""
        return 1
    fi
}

# Baseline - Debug build
measure_build "Debug (baseline)" ""

# Release builds
measure_build "ReleaseSafe" "-Doptimize=ReleaseSafe"
measure_build "ReleaseFast" "-Doptimize=ReleaseFast"
measure_build "ReleaseSmall" "-Doptimize=ReleaseSmall"

# Minimal build (no optional features)
measure_build "Minimal (ReleaseSmall, no AI/GPU/Web)" \
    "-Doptimize=ReleaseSmall -Denable-ai=false -Denable-gpu=false -Denable-web=false -Denable-monitoring=false"

# Database-only build
measure_build "Database-only (ReleaseSmall)" \
    "-Doptimize=ReleaseSmall -Denable-ai=false -Denable-gpu=false -Denable-web=false -Denable-monitoring=false"

echo "================================================"
echo "Build size comparison complete!"
echo "================================================"
echo ""
echo "Recommendations:"
echo "  - For production: zig build -Doptimize=ReleaseSafe"
echo "  - For max speed: zig build -Doptimize=ReleaseFast"  
echo "  - For min size: zig build -Doptimize=ReleaseSmall"
echo "  - For embedded: Add feature flags to disable unused features"
echo ""
