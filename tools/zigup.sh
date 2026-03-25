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
#   tools/zigup.sh --bootstrap  One-command project setup (prereqs + install + link)
#   tools/zigup.sh --doctor     Report toolchain health diagnostics

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ZIGVERSION_FILE="$REPO_ROOT/.zigversion"
CACHE_BASE="$HOME/.cache/abi-zig"

usage() {
    echo "Usage: zigup.sh [--status|--install|--update|--check|--clean|--link|--unlink|--bootstrap|--doctor]"
    echo "  --status    Print path to correct zig binary (install if missing)"
    echo "  --install   Force (re-)download and install zig + zls"
    echo "  --update    Check for newer zig and update if available"
    echo "  --check     Report if update available (no download)"
    echo "  --link      Symlink zig + zls into ~/.local/bin (or /usr/local/bin)"
    echo "  --unlink    Remove zig + zls symlinks from local bin"
    echo "  --clean     Remove all cached abi-zig versions"
    echo "  --bootstrap One-command project setup (prereqs + install + link + verify)"
    echo "  --doctor    Report toolchain health diagnostics"
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

resolve_zls_version() {
    # Query GitHub API to find the best ZLS release matching our zig major.minor.
    # Returns the tag name of the best match, or empty string on failure.
    if ! command -v python3 >/dev/null 2>&1; then
        echo "WARNING: python3 not found, cannot query GitHub API for ZLS version" >&2
        echo ""
        return 0
    fi

    # Extract major.minor from ZIG_VERSION (e.g. "0.16" from "0.16.0-dev.2979+e93834410")
    ZIG_MAJOR_MINOR="$(echo "$ZIG_VERSION" | sed 's/^\([0-9]*\.[0-9]*\).*/\1/')"

    RELEASES_JSON="$(curl -fsSL "https://api.github.com/repos/zigtools/zls/releases?per_page=10" 2>/dev/null)" || {
        echo ""
        return 0
    }

    echo "$RELEASES_JSON" | python3 -c "
import json, sys
releases = json.load(sys.stdin)
major_minor = '$ZIG_MAJOR_MINOR'
for r in releases:
    tag = r.get('tag_name', '')
    if tag.startswith(major_minor):
        print(tag)
        sys.exit(0)
# No match found
" 2>/dev/null || echo ""
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

    TARBALL="$CACHE_DIR/zls.tar.xz"
    ZLS_VERSION=""

    # Try to find the best ZLS version via GitHub API
    API_VERSION="$(resolve_zls_version)"
    if [ -n "$API_VERSION" ]; then
        URL="https://github.com/zigtools/zls/releases/download/${API_VERSION}/zls-${ZIG_OS}-${ZIG_ARCH}.tar.xz"
        echo "Trying ZLS $API_VERSION (from GitHub API) ..." >&2
        if try_download_url "$URL" "$TARBALL"; then
            ZLS_VERSION="$API_VERSION"
        fi
    fi

    # Fallback: try known versions if API query failed or download failed
    if [ -z "$ZLS_VERSION" ]; then
        echo "GitHub API lookup failed or no matching release, trying known versions ..." >&2
        for try_version in "0.16.0-dev.1" "0.15.0" "0.14.0"; do
            URL="https://github.com/zigtools/zls/releases/download/${try_version}/zls-${ZIG_OS}-${ZIG_ARCH}.tar.xz"
            echo "Trying ZLS $try_version ..." >&2
            if try_download_url "$URL" "$TARBALL"; then
                ZLS_VERSION="$try_version"
                break
            fi
        done
    fi

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

is_macos_26_4_plus() {
    # macOS 26.4+ corresponds to Darwin 25.x kernel
    [ "$(uname -s)" = "Darwin" ] || return 1
    DARWIN_MAJOR="$(uname -r | cut -d. -f1)"
    [ "$DARWIN_MAJOR" -ge 25 ] 2>/dev/null || return 1
}

do_bootstrap() {
    echo "=== ABI Toolchain Bootstrap ===" >&2
    echo "" >&2

    # Step 1: Check prerequisites
    echo "[1/6] Checking prerequisites ..." >&2
    PREREQ_OK=true

    if command -v curl >/dev/null 2>&1; then
        echo "  curl:    OK" >&2
    elif command -v wget >/dev/null 2>&1; then
        echo "  wget:    OK" >&2
    else
        echo "  curl/wget: MISSING (need one of these to download)" >&2
        PREREQ_OK=false
    fi

    if command -v tar >/dev/null 2>&1; then
        echo "  tar:     OK" >&2
    else
        echo "  tar:     MISSING" >&2
        PREREQ_OK=false
    fi

    if command -v python3 >/dev/null 2>&1; then
        echo "  python3: OK" >&2
    else
        echo "  python3: MISSING (needed for GitHub API queries and JSON parsing)" >&2
    fi

    # Step 2: macOS-specific checks
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "" >&2
        echo "[2/6] Checking macOS prerequisites ..." >&2
        if xcrun --version >/dev/null 2>&1; then
            echo "  Xcode CLI tools: OK" >&2
        else
            echo "  Xcode CLI tools: MISSING" >&2
            echo "  Run: xcode-select --install" >&2
            PREREQ_OK=false
        fi
    else
        echo "" >&2
        echo "[2/6] Skipping macOS checks (not on macOS)" >&2
    fi

    if [ "$PREREQ_OK" = false ]; then
        echo "" >&2
        echo "ERROR: Missing prerequisites. Install them and try again." >&2
        exit 1
    fi

    # Step 3: Install zig + zls
    echo "" >&2
    echo "[3/6] Installing zig + zls ..." >&2
    do_install >/dev/null

    # Step 4: Link to PATH
    echo "" >&2
    echo "[4/6] Linking to PATH ..." >&2
    do_link 2>&1 | while IFS= read -r line; do echo "  $line"; done >&2

    # Step 5: Verify installed version matches .zigversion
    echo "" >&2
    echo "[5/6] Verifying installation ..." >&2
    read_version
    detect_os
    detect_arch
    CACHE_DIR="$CACHE_BASE/$ZIG_VERSION"
    INSTALLED_VERSION=""
    if [ -x "$CACHE_DIR/bin/zig" ]; then
        INSTALLED_VERSION="$("$CACHE_DIR/bin/zig" version 2>/dev/null || echo "")"
    fi
    if [ "$INSTALLED_VERSION" = "$ZIG_VERSION" ]; then
        echo "  zig version: $INSTALLED_VERSION (matches .zigversion)" >&2
    elif [ -n "$INSTALLED_VERSION" ]; then
        echo "  WARNING: zig version $INSTALLED_VERSION does not match .zigversion ($ZIG_VERSION)" >&2
    else
        echo "  WARNING: could not verify zig version" >&2
    fi

    # Step 6: Platform-specific notes
    echo "" >&2
    echo "[6/6] Platform notes ..." >&2
    if is_macos_26_4_plus; then
        echo "  NOTE: macOS 26.4+ detected (Darwin $(uname -r))." >&2
        echo "  Stock Zig's LLD linker cannot link on this OS version." >&2
        echo "  Use ./build.sh instead of zig build for all build commands." >&2
    else
        echo "  Standard platform. zig build works directly." >&2
    fi

    # Success summary
    echo "" >&2
    echo "=== Bootstrap Complete ===" >&2
    echo "" >&2
    echo "Next steps:" >&2
    echo "  1. Ensure ~/.local/bin is on your PATH:" >&2
    echo "       export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    echo "  2. Build the project:" >&2
    if is_macos_26_4_plus; then
        echo "       ./build.sh                      # build" >&2
        echo "       ./build.sh test --summary all   # test" >&2
    else
        echo "       zig build                       # build" >&2
        echo "       zig build test --summary all    # test" >&2
    fi
    echo "  3. Run the CLI:" >&2
    if is_macos_26_4_plus; then
        echo "       ./build.sh cli && zig-out/bin/abi doctor" >&2
    else
        echo "       zig build cli && zig-out/bin/abi doctor" >&2
    fi
    echo "" >&2
}

do_doctor() {
    echo "=== ABI Toolchain Doctor ===" >&2
    echo "" >&2

    # 1. .zigversion value
    read_version
    detect_os
    detect_arch
    CACHE_DIR="$CACHE_BASE/$ZIG_VERSION"
    echo ".zigversion:  $ZIG_VERSION" >&2

    # 2. zig binary path + version
    ZIG_PATH=""
    ZIG_STATUS="MISSING"
    # Check cache first
    if [ -x "$CACHE_DIR/bin/zig" ]; then
        ZIG_PATH="$CACHE_DIR/bin/zig"
    elif command -v zig >/dev/null 2>&1; then
        ZIG_PATH="$(command -v zig)"
    fi

    if [ -n "$ZIG_PATH" ]; then
        ACTUAL_VERSION="$("$ZIG_PATH" version 2>/dev/null || echo "unknown")"
        if [ "$ACTUAL_VERSION" = "$ZIG_VERSION" ]; then
            ZIG_STATUS="OK"
        else
            ZIG_STATUS="MISMATCH (got $ACTUAL_VERSION)"
        fi
    fi
    echo "zig binary:   ${ZIG_PATH:-not found}  [$ZIG_STATUS]" >&2

    # 3. zls binary path + version
    ZLS_PATH=""
    ZLS_STATUS="MISSING"
    if [ -x "$CACHE_DIR/bin/zls" ]; then
        ZLS_PATH="$CACHE_DIR/bin/zls"
    elif command -v zls >/dev/null 2>&1; then
        ZLS_PATH="$(command -v zls)"
    fi

    if [ -n "$ZLS_PATH" ]; then
        ZLS_ACTUAL="$("$ZLS_PATH" --version 2>/dev/null || echo "unknown")"
        ZLS_STATUS="OK ($ZLS_ACTUAL)"
    fi
    echo "zls binary:   ${ZLS_PATH:-not found}  [$ZLS_STATUS]" >&2

    # 4. PATH check
    if command -v zig >/dev/null 2>&1; then
        echo "zig on PATH:  YES ($(command -v zig))" >&2
    else
        echo "zig on PATH:  NO" >&2
        echo "  Fix: tools/zigup.sh --link && export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    fi

    # 5. Platform
    echo "platform:     $(uname -s)/$(uname -m)" >&2

    # 6. macOS: Xcode CLI tools
    if [ "$(uname -s)" = "Darwin" ]; then
        if xcrun --version >/dev/null 2>&1; then
            echo "xcode-cli:    INSTALLED" >&2
        else
            echo "xcode-cli:    MISSING" >&2
            echo "  Fix: xcode-select --install" >&2
        fi

        # 7. macOS 26.4+ LLD workaround
        if is_macos_26_4_plus; then
            echo "macos 26.4+:  YES -- use ./build.sh (stock LLD fails)" >&2
        else
            echo "macos 26.4+:  NO (zig build works directly)" >&2
        fi
    fi

    # 8. Quick build check
    echo "" >&2
    echo "Build check:" >&2
    if [ -n "$ZIG_PATH" ]; then
        if is_macos_26_4_plus && [ -x "$REPO_ROOT/build.sh" ]; then
            BUILD_CMD="$REPO_ROOT/build.sh"
            BUILD_LABEL="./build.sh doctor"
        else
            BUILD_CMD="$ZIG_PATH build"
            BUILD_LABEL="zig build doctor"
        fi
        if (cd "$REPO_ROOT" && $BUILD_CMD doctor >/dev/null 2>&1); then
            echo "$BUILD_LABEL: OK" >&2
        else
            echo "$BUILD_LABEL: FAIL (run manually to see errors)" >&2
        fi
    else
        echo "zig build doctor: SKIP (zig not found)" >&2
    fi

    echo "" >&2
}

case "${1:-}" in
    --status)    do_status ;;
    --install)   do_install ;;
    --update)    do_update ;;
    --check)     do_check ;;
    --link)      do_link ;;
    --unlink)    do_unlink ;;
    --clean)     do_clean ;;
    --bootstrap) do_bootstrap ;;
    --doctor)    do_doctor ;;
    *)           usage ;;
esac
