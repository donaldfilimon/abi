#!/bin/bash

# Zig Release Monitor Script
# Monitors Zig releases and updates project configuration accordingly

set -e

echo "🔍 Zig Release Monitor"
echo "====================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GITHUB_API_URL="https://api.github.com/repos/ziglang/zig/releases"
CURRENT_VERSION_FILE=".zigversion"
CI_WORKFLOW_FILE=".github/workflows/ci.yml"
DOCS_WORKFLOW_FILE=".github/workflows/deploy_docs.yml"

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

# Get latest releases from GitHub API
get_latest_releases() {
    print_status "Fetching latest Zig releases from GitHub..."

    if ! curl -s "$GITHUB_API_URL" > /tmp/zig_releases.json; then
        print_error "Failed to fetch releases from GitHub API"
        return 1
    fi

    print_success "Successfully fetched release information"
}

# Extract version information
extract_versions() {
    print_status "Analyzing release versions..."

    # Get latest stable release (if any)
    LATEST_STABLE=$(jq -r '.[] | select(.prerelease == false) | .tag_name' /tmp/zig_releases.json | head -1)

    # Get latest dev release
    LATEST_DEV=$(jq -r '.[0].tag_name' /tmp/zig_releases.json)

    # Get current version from project
    if [ -f "$CURRENT_VERSION_FILE" ]; then
        CURRENT_VERSION=$(cat "$CURRENT_VERSION_FILE" | tr -d '\n')
        print_status "Current project version: $CURRENT_VERSION"
    else
        print_warning "No current version file found"
        CURRENT_VERSION=""
    fi

    print_success "Latest stable: ${LATEST_STABLE:-None}"
    print_success "Latest dev: $LATEST_DEV"
}

# Update project files if needed
update_project_files() {
    local new_version="$1"
    local is_stable="$2"

    print_status "Checking if project needs updates..."

    if [ "$CURRENT_VERSION" = "$new_version" ]; then
        print_success "Project is already up to date"
        return 0
    fi

    print_warning "Project version mismatch detected"
    print_status "Current: $CURRENT_VERSION"
    print_status "Latest: $new_version"

    # Update .zigversion file
    echo "$new_version" > "$CURRENT_VERSION_FILE"
    print_success "Updated $CURRENT_VERSION_FILE"

    # Update CI workflow
    if [ -f "$CI_WORKFLOW_FILE" ]; then
        # Add new version to matrix, keep old ones for compatibility
        sed -i.bak "s/0\.16\.0-dev\.[0-9]\+/&,$new_version/" "$CI_WORKFLOW_FILE"
        print_success "Updated CI workflow with new version"
    fi

    # Update docs workflow
    if [ -f "$DOCS_WORKFLOW_FILE" ]; then
        sed -i.bak "s/version: 0\.16\.0-dev\.[0-9]\+/version: $new_version/" "$DOCS_WORKFLOW_FILE"
        print_success "Updated docs workflow with new version"
    fi

    # Create a summary
    echo "# Zig Version Update - $(date)" > /tmp/zig_update_summary.md
    echo "" >> /tmp/zig_update_summary.md
    echo "## Changes Made" >> /tmp/zig_update_summary.md
    echo "- Updated $CURRENT_VERSION_FILE to $new_version" >> /tmp/zig_update_summary.md
    echo "- Updated CI pipeline to include $new_version" >> /tmp/zig_update_summary.md
    echo "- Updated documentation workflow to use $new_version" >> /tmp/zig_update_summary.md
    echo "" >> /tmp/zig_update_summary.md
    echo "## Version Details" >> /tmp/zig_update_summary.md
    echo "- Previous version: $CURRENT_VERSION" >> /tmp/zig_update_summary.md
    echo "- New version: $new_version" >> /tmp/zig_update_summary.md
    echo "- Stable release: $(if [ "$is_stable" = "true" ]; then echo "Yes"; else echo "No"; fi)" >> /tmp/zig_update_summary.md

    print_success "Update summary created"
}

# Test build with new version
test_build() {
    local version="$1"

    print_status "Testing build with Zig $version..."

    if ! command -v zig &> /dev/null; then
        print_warning "Zig not found in PATH, skipping build test"
        return 0
    fi

    # Test basic build
    if zig build --version | grep -q "$version"; then
        print_success "Zig version matches expected: $version"
    else
        print_warning "Zig version mismatch, but continuing..."
    fi

    # Test project build
    if zig build; then
        print_success "Project builds successfully with $version"
    else
        print_error "Project build failed with $version"
        return 1
    fi
}

# Main execution
main() {
    get_latest_releases
    extract_versions

    # Prefer stable release if available, otherwise use latest dev
    if [ -n "$LATEST_STABLE" ]; then
        print_status "Stable release available: $LATEST_STABLE"
        update_project_files "$LATEST_STABLE" "true"
        test_build "$LATEST_STABLE"
    else
        print_status "No stable release, using latest dev: $LATEST_DEV"
        update_project_files "$LATEST_DEV" "false"
        test_build "$LATEST_DEV"
    fi

    # Show summary
    if [ -f /tmp/zig_update_summary.md ]; then
        echo ""
        echo "📋 Update Summary:"
        echo "=================="
        cat /tmp/zig_update_summary.md
    fi

    print_success "Zig release monitoring completed"
}

# Run main function
main "$@"
