#!/bin/bash
# Zig Version Monitor Script
# Usage: ./scripts/check-zig-version.sh [--json]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Zig Version Monitor"
echo "========================================"
echo ""

# Get current minimum version from build.zig.zon
CURRENT_VERSION=$(grep 'minimum_zig_version' build.zig.zon 2>/dev/null | sed 's/.*"\(.*\)".*/\1/')
if [ -z "$CURRENT_VERSION" ]; then
    echo -e "${RED}Error: Could not read minimum_zig_version from build.zig.zon${NC}"
    exit 1
fi
echo -e "${YELLOW}Current minimum version:${NC} $CURRENT_VERSION"

# Get installed Zig version
INSTALLED_VERSION=$(zig version 2>/dev/null)
echo -e "${YELLOW}Installed Zig version:${NC} $INSTALLED_VERSION"

# Get latest stable Zig version (using simpler grep patterns)
echo ""
echo "Checking for latest stable release..."
LATEST_STABLE=$(curl -s https://ziglang.org/download/ 2>/dev/null | \
    grep "release-" | head -1 | sed -n 's/.*release-\([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/p')

if [ -z "$LATEST_STABLE" ]; then
    echo -e "${RED}Warning: Could not fetch latest version from ziglang.org${NC}"
    LATEST_STABLE="unknown"
else
    echo -e "${GREEN}Latest stable release:${NC} $LATEST_STABLE"
fi

# Version comparison
echo ""
echo "========================================"
echo "  Status Check"
echo "========================================"

# Simple version comparison
extract_major() { echo "$1" | cut -d. -f1; }
extract_minor() { echo "$1" | cut -d. -f2; }
extract_patch() { echo "$1" | cut -d. -f3 | tr -d 'a-z' | tr -d 'A-Z'; }

CURRENT_BASE=$(echo "$CURRENT_VERSION" | sed 's/-dev\..*//')
CURRENT_BASE=$(echo "$CURRENT_BASE" | sed 's/+.*//')

# Check if versions match
if [ "$LATEST_STABLE" = "$CURRENT_BASE" ]; then
    echo -e "${GREEN}✓ Current${NC}"
    echo "  Already at latest stable version."
    UPDATE_AVAILABLE="false"
elif [ -n "$LATEST_STABLE" ] && [ "$LATEST_STABLE" != "unknown" ]; then
    # Compare versions
    C_MAJ=$(extract_major "$CURRENT_BASE")
    C_MIN=$(extract_minor "$CURRENT_BASE")
    C_PAT=$(extract_patch "$CURRENT_BASE")
    L_MAJ=$(extract_major "$LATEST_STABLE")
    L_MIN=$(extract_minor "$LATEST_STABLE")
    L_PAT=$(extract_patch "$LATEST_STABLE")

    if [ "$L_MAJ" -gt "$C_MAJ" ] || \
       [ "$L_MAJ" -eq "$C_MAJ" ] && [ "$L_MIN" -gt "$C_MIN" ] || \
       [ "$L_MAJ" -eq "$C_MAJ" ] && [ "$L_MIN" -eq "$C_MIN" ] && [ "$L_PAT" -gt "$C_PAT" ]; then
        echo -e "${YELLOW}⚠ Update Recommended${NC}"
        echo "  A newer stable release ($LATEST_STABLE) is available."
        echo "  Consider updating minimum_zig_version in build.zig.zon"
        UPDATE_AVAILABLE="true"
    else
        echo -e "${GREEN}✓ Up to date${NC}"
        echo "  Current minimum meets or exceeds latest stable."
        UPDATE_AVAILABLE="false"
    fi
else
    echo -e "${GREEN}✓ Current${NC}"
    UPDATE_AVAILABLE="false"
fi

# JSON output for scripts
if [ "$1" = "--json" ]; then
    echo ""
    echo "========================================"
    echo "  JSON Output"
    echo "========================================"
    cat <<EOF
{
  "current_minimum": "$CURRENT_VERSION",
  "installed": "$INSTALLED_VERSION",
  "latest_stable": "$LATEST_STABLE",
  "update_available": $UPDATE_AVAILABLE
}
EOF
fi

echo ""
echo "========================================"
echo "  Recommendations"
echo "========================================"
echo "1. Update CI: Edit .github/workflows/ci.yml"
echo "2. Update minimum: Edit build.zig.zon"
echo "3. Test changes: zig build test --summary all"
echo "4. Update CHANGELOG.md with new version"
echo "========================================"
