#!/bin/sh
set -eu

# ABI Auto-Update System
# Checks for new zig dev builds, verifies, and commits version bumps.
#
# Usage:
#   tools/auto_update.sh --check    Report if update available
#   tools/auto_update.sh --update   Update and verify
#   tools/auto_update.sh --auto     Full auto: check, update, verify, commit

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZIGVERSION_FILE="$REPO_ROOT/.zigversion"
ZIGLY="$SCRIPT_DIR/zigly"
DOWNLOAD_INDEX="https://ziglang.org/download/index.json"

usage() {
    echo "Usage: auto_update.sh [--check|--update|--auto]"
    echo "  --check   Report if update is available (no changes)"
    echo "  --update  Update .zigversion, run checks"
    echo "  --auto    Full auto: check, update, verify, commit"
    exit 1
}

log() { echo "[auto-update] $*" >&2; }
err() { echo "[auto-update] ERROR: $*" >&2; exit 1; }

read_current_version() {
    if [ ! -f "$ZIGVERSION_FILE" ]; then
        err "$ZIGVERSION_FILE not found"
    fi
    CURRENT_VERSION="$(cat "$ZIGVERSION_FILE" | tr -d '[:space:]')"
    if [ -z "$CURRENT_VERSION" ]; then
        err ".zigversion is empty"
    fi
}

detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        *)      err "Unsupported OS: $(uname -s)" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        arm64|aarch64)  echo "aarch64" ;;
        x86_64|amd64)  echo "x86_64" ;;
        *)              err "Unsupported arch: $(uname -m)" ;;
    esac
}

fetch_json_field() {
    # $1 = URL, $2 = field path
    if command -v python3 >/dev/null 2>&1; then
        curl -fsSL "$1" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
keys='$2'.split('.')
v=d
for k in keys:
    if k.isdigit(): v=v[int(k)]
    else: v=v[k]
print(v)
" 2>/dev/null || true
    else
        echo ""
    fi
}

get_latest_version() {
    # Fetch the latest master version string from zig download index
    LATEST_VERSION="$(fetch_json_field "$DOWNLOAD_INDEX" "master.version")"
    if [ -z "$LATEST_VERSION" ]; then
        err "Could not fetch latest version from $DOWNLOAD_INDEX"
    fi
}

get_latest_tarball_url() {
    os="$(detect_os)"
    arch="$(detect_arch)"
    platform="${arch}-${os}"
    LATEST_URL="$(fetch_json_field "$DOWNLOAD_INDEX" "master.${platform}.tarball")"
}

# Compare version strings. Returns 0 if $1 != $2
versions_differ() {
    [ "$1" != "$2" ]
}

do_check() {
    read_current_version
    get_latest_version
    log "Current version: $CURRENT_VERSION"
    log "Latest version:  $LATEST_VERSION"

    if versions_differ "$CURRENT_VERSION" "$LATEST_VERSION"; then
        log "Update available: $CURRENT_VERSION -> $LATEST_VERSION"
        get_latest_tarball_url
        log "Tarball: ${LATEST_URL:-<unavailable>}"
        return 0
    else
        log "Already up to date."
        return 1
    fi
}

do_update() {
    read_current_version
    get_latest_version

    if ! versions_differ "$CURRENT_VERSION" "$LATEST_VERSION"; then
        log "Already up to date: $CURRENT_VERSION"
        return 0
    fi

    log "Updating: $CURRENT_VERSION -> $LATEST_VERSION"

    # Backup current version
    cp "$ZIGVERSION_FILE" "${ZIGVERSION_FILE}.bak"

    # Write new version
    echo "$LATEST_VERSION" > "$ZIGVERSION_FILE"
    log "Updated .zigversion to $LATEST_VERSION"

    # Force reinstall zig
    log "Installing zig $LATEST_VERSION ..."
    "$ZIGLY" --install || {
        log "Install failed, reverting .zigversion"
        mv "${ZIGVERSION_FILE}.bak" "$ZIGVERSION_FILE"
        err "zig install failed for $LATEST_VERSION"
    }

    # Run build check
    log "Running 'zig build check' to verify ..."
    ZIG="$("$ZIGLY" --status)"
    ZIG_LIB="$(dirname "$(dirname "$ZIG")")/lib"
    if "$ZIG" build check --zig-lib-dir "$ZIG_LIB" --global-cache-dir "$HOME/.cache/zig" --cache-dir .zig-cache 2>&1; then
        log "Check passed!"
        rm -f "${ZIGVERSION_FILE}.bak"
        return 0
    else
        log "Check failed! Reverting to $CURRENT_VERSION"
        mv "${ZIGVERSION_FILE}.bak" "$ZIGVERSION_FILE"
        "$ZIGLY" --install
        err "Build check failed with zig $LATEST_VERSION, reverted to $CURRENT_VERSION"
    fi
}

do_auto() {
    # Check first
    if ! do_check; then
        log "No update needed."
        return 0
    fi

    # Update and verify
    do_update

    # Commit the version bump
    if command -v git >/dev/null 2>&1 && [ -d "$REPO_ROOT/.git" ]; then
        cd "$REPO_ROOT"
        git add .zigversion
        if ! git diff --cached --quiet .zigversion; then
            git commit -m "chore: update zig to $LATEST_VERSION

Updated .zigversion from $CURRENT_VERSION to $LATEST_VERSION.
Verified with 'zig build check'."
            log "Committed version bump: $CURRENT_VERSION -> $LATEST_VERSION"
        else
            log "No changes to commit (version unchanged after install)."
        fi
    else
        log "Not a git repo, skipping commit."
    fi
}

# Main
case "${1:-}" in
    --check)  do_check ;;
    --update) do_update ;;
    --auto)   do_auto ;;
    *)        usage ;;
esac
