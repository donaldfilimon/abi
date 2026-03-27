#!/bin/sh
set -eu

# ABI Zig Version Manager
# Reads .zigversion, downloads matching zig + ZLS to ~/.cache/abi-zig/
#
# Usage:
#   tools/zigup.sh --status     Print zig path (install if needed)
#   tools/zigup.sh --install    Force (re-)install zig + zls
#   tools/zigup.sh --update     Check for newer zig and update if available
#   tools/zigup.sh --check      Report if update available (no download)
#   tools/zigup.sh --clean      Remove cached downloads

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZIGVERSION_FILE="$REPO_ROOT/.zigversion"
CACHE_BASE="$HOME/.cache/abi-zig"

usage() {
    echo "Usage: zigup.sh [--status|--install|--update|--check|--clean|--link|--unlink]"
    echo "  --status   Print path to correct zig binary (install if missing)"
    echo "  --install  Force (re-)download and install zig + zls"
    echo "  --update   Check for newer zig and update if available"
    echo "  --check    Report if update available (no download)"
    echo "  --link     Symlink zig + zls into ~/.local/bin (or /usr/local/bin)"
    echo "  --unlink   Remove zig + zls symlinks from local bin"
    echo "  --clean    Remove all cached abi-zig versions"
    exit 1
}

read_version() {
    if [ ! -f "$ZIGVERSION_FILE" ]; then
        echo "ERROR: $ZIGVERSION_FILE not found" >&2
        exit 1
    fi
    ZIG_VERSION="$(cat "$ZIGVERSION_FILE" | tr -d '[:space:]')"
    if [ -z "$ZIG_VERSION" ]; then
        echo "ERROR: .zigversion is empty" >&2
        exit 1
    fi
}

detect_os() {
    case "$(uname -s)" in
        Darwin) ZIG_OS="macos" ;;
        Linux)  ZIG_OS="linux" ;;
        *)      echo "ERROR: Unsupported OS: $(uname -s)" >&2; exit 1 ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        arm64|aarch64)  ZIG_ARCH="aarch64" ;;
        x86_64|amd64)  ZIG_ARCH="x86_64" ;;
        i386|i686)      ZIG_ARCH="x86" ;;
        *)              echo "ERROR: Unsupported arch: $(uname -m)" >&2; exit 1 ;;
    esac
}

check_zig_installed() {
    ZIG_BIN="$CACHE_DIR/bin/zig"
    # If the binary exists and a metadata file marks it as provisioned, we're good.
    if [ -x "$ZIG_BIN" ] && [ -f "$CACHE_DIR/.zigup_installed" ]; then
        return 0
    fi
    return 1
}

check_zls_installed() {
    ZLS_BIN="$CACHE_DIR/bin/zls"
    [ -x "$ZLS_BIN" ]
}

try_download_url() {
    # $1 = URL, $2 = output path. Returns 0 on success.
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$2" "$1" 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1" 2>/dev/null
    else
        echo "ERROR: Need curl or wget to download" >&2
        exit 1
    fi
}

fetch_json_field() {
    # $1 = URL, $2 = field path (python jq-like). Returns value or empty.
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
        echo "" >&2
    fi
}

resolve_master_tarball_url() {
    # Query zig download API to get the actual master tarball URL
    PLATFORM_KEY="${ZIG_ARCH}-${ZIG_OS}"
    fetch_json_field "https://ziglang.org/download/index.json" "master.${PLATFORM_KEY}.tarball"
}

download_zig() {
    TARBALL="$CACHE_DIR/zig.tar.xz"
    EXTRACT_DIR="$CACHE_DIR/_extract"
    mkdir -p "$CACHE_DIR" "$EXTRACT_DIR"

    # URL pattern: zig-{arch}-{os}-{version}.tar.xz
    DIRECT_URL="https://ziglang.org/builds/zig-${ZIG_ARCH}-${ZIG_OS}-${ZIG_VERSION}.tar.xz"

    echo "Trying: $DIRECT_URL" >&2
    if try_download_url "$DIRECT_URL" "$TARBALL"; then
        echo "Downloaded zig $ZIG_VERSION" >&2
    else
        # Dev version not on server. Query the download API for latest master.
        echo "Exact version not available, fetching latest master ..." >&2
        MASTER_URL="$(resolve_master_tarball_url)"
        if [ -z "$MASTER_URL" ]; then
            echo "ERROR: Could not resolve master download URL from ziglang.org API" >&2
            rm -rf "$EXTRACT_DIR"
            exit 1
        fi
        echo "Trying: $MASTER_URL" >&2
        if try_download_url "$MASTER_URL" "$TARBALL"; then
            echo "Downloaded latest zig master" >&2
        else
            echo "ERROR: Could not download zig from $MASTER_URL" >&2
            rm -rf "$EXTRACT_DIR"
            exit 1
        fi
    fi

    echo "Extracting ..." >&2
    tar -xf "$TARBALL" -C "$EXTRACT_DIR"
    INNER_DIR=$(ls -1d "$EXTRACT_DIR"/zig-* 2>/dev/null | head -1)
    if [ -z "$INNER_DIR" ] || [ ! -d "$INNER_DIR" ]; then
        echo "ERROR: Unexpected archive structure" >&2
        exit 1
    fi
    mkdir -p "$CACHE_DIR/bin"
    cp "$INNER_DIR/zig" "$CACHE_DIR/bin/zig" 2>/dev/null || true
    cp "$INNER_DIR/zig.exe" "$CACHE_DIR/bin/zig.exe" 2>/dev/null || true
    chmod +x "$CACHE_DIR/bin/zig" 2>/dev/null || true
    # Copy lib/ needed for zig build
    if [ -d "$INNER_DIR/lib" ]; then
        cp -R "$INNER_DIR/lib" "$CACHE_DIR/lib"
    fi
    rm -rf "$EXTRACT_DIR" "$TARBALL"
    echo "zig installed to $CACHE_DIR/bin/zig" >&2
    echo "$ZIG_VERSION" > "$CACHE_DIR/.zigup_installed"
}

download_zls() {
    # If zls already exists in cache, skip
    if [ -x "$CACHE_DIR/bin/zls" ]; then
        echo "ZLS already installed" >&2
        return 0
    fi

    # Prefer zvm's ZLS if available (works for dev zig versions)
    ZVM_ZLS="$HOME/.zvm/bin/zls"
    if [ -x "$ZVM_ZLS" ]; then
        echo "Copying ZLS from zvm ..." >&2
        cp "$ZVM_ZLS" "$CACHE_DIR/bin/zls"
        chmod +x "$CACHE_DIR/bin/zls"
        echo "ZLS installed from zvm to $CACHE_DIR/bin/zls" >&2
        return 0
    fi

    # Try known ZLS release URLs
    TARBALL="$CACHE_DIR/zls.tar.xz"
    ZLS_VERSION=""
    for try_version in "0.16.0-dev.1" "0.15.0" "0.14.0"; do
        URL="https://github.com/zigtools/zls/releases/download/${try_version}/zls-${ZIG_OS}-${ZIG_ARCH}.tar.xz"
        echo "Trying ZLS $try_version ..." >&2
        if try_download_url "$URL" "$TARBALL"; then
            ZLS_VERSION="$try_version"
            break
        fi
    done

    if [ -z "$ZLS_VERSION" ]; then
        echo "WARNING: No pre-built ZLS found." >&2
        echo "  Install via: zvm install master  (provides zls)" >&2
        echo "  Or build from source: git clone --depth 1 https://github.com/zigtools/zls && cd zls && zig build -Doptimize=ReleaseSafe" >&2
        rm -f "$TARBALL"
        return 0
    fi

    echo "Extracting ZLS $ZLS_VERSION ..." >&2
    ZLS_EXTRACT_DIR="$CACHE_DIR/_zls_extract"
    mkdir -p "$ZLS_EXTRACT_DIR"
    tar -xf "$TARBALL" -C "$ZLS_EXTRACT_DIR"
    ZLS_FOUND=$(find "$ZLS_EXTRACT_DIR" -name "zls" -type f 2>/dev/null | head -1)
    if [ -n "$ZLS_FOUND" ]; then
        cp "$ZLS_FOUND" "$CACHE_DIR/bin/zls"
        chmod +x "$CACHE_DIR/bin/zls"
        echo "ZLS $ZLS_VERSION installed to $CACHE_DIR/bin/zls" >&2
    else
        echo "WARNING: zls binary not found in archive" >&2
    fi
    rm -rf "$ZLS_EXTRACT_DIR" "$TARBALL"
}

do_install() {
    read_version
    detect_os
    detect_arch
    CACHE_DIR="$CACHE_BASE/$ZIG_VERSION"
    download_zig
    download_zls
    echo "$CACHE_DIR/bin/zig"
}

do_status() {
    read_version
    detect_os
    detect_arch
    CACHE_DIR="$CACHE_BASE/$ZIG_VERSION"
    if ! check_zig_installed; then
        echo "zig $ZIG_VERSION not installed, downloading ..." >&2
        do_install
        return
    fi
    echo "$CACHE_DIR/bin/zig"
}

do_clean() {
    if [ -d "$CACHE_BASE" ]; then
        echo "Removing $CACHE_BASE ..." >&2
        rm -rf "$CACHE_BASE"
        echo "Cleaned." >&2
    else
        echo "Nothing to clean." >&2
    fi
}

get_remote_version() {
    # Fetch the latest master version from zig download index
    fetch_json_field "https://ziglang.org/download/index.json" "master.version"
}

do_check() {
    read_version
    REMOTE_VERSION="$(get_remote_version)"
    if [ -z "$REMOTE_VERSION" ]; then
        echo "ERROR: Could not fetch latest version from ziglang.org" >&2
        exit 1
    fi
    echo "Current: $ZIG_VERSION"
    echo "Latest:  $REMOTE_VERSION"
    if [ "$ZIG_VERSION" = "$REMOTE_VERSION" ]; then
        echo "Already up to date."
        return 1
    else
        echo "Update available: $ZIG_VERSION -> $REMOTE_VERSION"
        return 0
    fi
}

do_update() {
    read_version
    REMOTE_VERSION="$(get_remote_version)"
    if [ -z "$REMOTE_VERSION" ]; then
        echo "ERROR: Could not fetch latest version from ziglang.org" >&2
        exit 1
    fi
    if [ "$ZIG_VERSION" = "$REMOTE_VERSION" ]; then
        echo "Already up to date: $ZIG_VERSION" >&2
        return 0
    fi
    echo "Updating: $ZIG_VERSION -> $REMOTE_VERSION" >&2
    # Update .zigversion and reinstall
    echo "$REMOTE_VERSION" > "$ZIGVERSION_FILE"
    ZIG_VERSION="$REMOTE_VERSION"
    CACHE_DIR="$CACHE_BASE/$ZIG_VERSION"
    download_zig
    download_zls
    echo "Updated to $REMOTE_VERSION" >&2
    echo "$CACHE_DIR/bin/zig"
}

# Determine the best local bin directory for user installs
resolve_local_bin() {
    # Prefer ~/.local/bin (XDG standard, always writable)
    if [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin" 2>/dev/null; then
        echo "$HOME/.local/bin"
        return 0
    fi
    # Fallback to /usr/local/bin if writable
    if [ -w "/usr/local/bin" ]; then
        echo "/usr/local/bin"
        return 0
    fi
    echo "ERROR: Cannot find writable local bin directory" >&2
    echo "  Create ~/.local/bin and ensure it is on your PATH" >&2
    return 1
}

do_link() {
    read_version
    detect_os
    detect_arch
    CACHE_DIR="$CACHE_BASE/$ZIG_VERSION"

    # Ensure zig is installed in cache
    if ! check_zig_installed; then
        echo "zig not in cache, installing first ..." >&2
        download_zig
        download_zls
    fi

    LOCAL_BIN="$(resolve_local_bin)" || exit 1

    echo "Linking zig + zls to $LOCAL_BIN ..." >&2

    # Symlink zig
    if [ -x "$CACHE_DIR/bin/zig" ]; then
        ln -sf "$CACHE_DIR/bin/zig" "$LOCAL_BIN/zig"
        echo "  zig -> $LOCAL_BIN/zig" >&2
    else
        echo "WARNING: zig binary not found in cache" >&2
    fi

    # Symlink zls
    if [ -x "$CACHE_DIR/bin/zls" ]; then
        ln -sf "$CACHE_DIR/bin/zls" "$LOCAL_BIN/zls"
        echo "  zls -> $LOCAL_BIN/zls" >&2
    else
        echo "WARNING: zls binary not found in cache" >&2
    fi

    # Verify PATH includes local bin
    case ":$PATH:" in
        *":$LOCAL_BIN:"*)
            echo "Done. $LOCAL_BIN is on PATH." >&2
            ;;
        *)
            echo "" >&2
            echo "WARNING: $LOCAL_BIN is not on your PATH." >&2
            echo "Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):" >&2
            echo "" >&2
            echo "  export PATH=\"$LOCAL_BIN:\$PATH\"" >&2
            echo "" >&2
            ;;
    esac

    # Show versions
    echo "" >&2
    "$LOCAL_BIN/zig" version 2>/dev/null && true
    if [ -x "$LOCAL_BIN/zls" ]; then
        "$LOCAL_BIN/zls" --version 2>/dev/null && true
    fi
}

do_unlink() {
    LOCAL_BIN="$(resolve_local_bin)" || exit 1

    echo "Removing zig + zls symlinks from $LOCAL_BIN ..." >&2

    if [ -L "$LOCAL_BIN/zig" ]; then
        rm "$LOCAL_BIN/zig"
        echo "  Removed $LOCAL_BIN/zig" >&2
    fi

    if [ -L "$LOCAL_BIN/zls" ]; then
        rm "$LOCAL_BIN/zls"
        echo "  Removed $LOCAL_BIN/zls" >&2
    fi

    echo "Done." >&2
}

case "${1:-}" in
    --status)  do_status ;;
    --install) do_install ;;
    --update)  do_update ;;
    --check)   do_check ;;
    --link)    do_link ;;
    --unlink)  do_unlink ;;
    --clean)   do_clean ;;
    *)         usage ;;
esac
