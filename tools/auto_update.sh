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

## JSON field extraction — no Python3 required.
## Parses the zig download index using grep/sed. The index structure is flat
## enough that we can extract "version" and "tarball" with line-based matching.
_INDEX_CACHE=""

fetch_index() {
    if [ -z "$_INDEX_CACHE" ]; then
        _INDEX_CACHE="$(curl -fsSL "$DOWNLOAD_INDEX" 2>/dev/null)" || true
    fi
}

# Extract a quoted string value for a given key from the cached index JSON.
# Works for simple "key": "value" pairs (covers version and tarball fields).
json_string_value() {
    local key="$1"
    printf '%s\n' "$_INDEX_CACHE" | grep "\"${key}\"" | head -1 | sed -E 's/.*"'"${key}"'"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/'
}

get_latest_version() {
    fetch_index
    LATEST_VERSION="$(json_string_value "version")"
    if [ -z "$LATEST_VERSION" ]; then
        err "Could not fetch latest version from $DOWNLOAD_INDEX"
    fi
}

get_latest_tarball_url() {
    fetch_index
    os="$(detect_os)"
    arch="$(detect_arch)"
    platform="${arch}-${os}"
    # The tarball URL appears inside the platform-specific block; grep for it
    # after narrowing to lines near the platform key.
    LATEST_URL="$(printf '%s\n' "$_INDEX_CACHE" | sed -n "/\"${platform}\"/,/}/p" | grep '"tarball"' | head -1 | sed -E 's/.*"tarball"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"
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

<<<<<<< Updated upstream
    # Force reinstall zig through the shared zigly/zvm resolution path
=======
    if command -v zvm >/dev/null 2>&1; then
        log "Updating zvm master with ZLS..."
        zvm i --zls master || log "zvm update failed, continuing..."
    fi

    # Force reinstall zig
>>>>>>> Stashed changes
    log "Installing zig $LATEST_VERSION ..."
    "$ZIGLY" --install "$LATEST_VERSION" || {
        log "Install failed, reverting .zigversion"
        mv "${ZIGVERSION_FILE}.bak" "$ZIGVERSION_FILE"
        err "zig install failed for $LATEST_VERSION"
    }

    # Run the repo wrapper check so Darwin toolchain handling stays truthful
    log "Running './build.sh check --summary all' to verify ..."
    if (
        cd "$REPO_ROOT"
        ./build.sh check --summary all
    ) 2>&1; then
        log "Check passed!"
        rm -f "${ZIGVERSION_FILE}.bak"
        return 0
    else
        log "Check failed! Reverting to $CURRENT_VERSION"
        mv "${ZIGVERSION_FILE}.bak" "$ZIGVERSION_FILE"
        "$ZIGLY" --install "$CURRENT_VERSION"
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
Verified with './build.sh check --summary all'."
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
