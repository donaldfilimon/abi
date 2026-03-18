#!/bin/bash
# run_build.sh — Direct build runner invocation for macOS 26+
#
# Simple two-pass script:
#   1. zig build → fails at link → build_zcu.o exists
#   2. Relink .o with Apple's /usr/bin/ld → build runner binary
#   3. Execute build runner directly with the right args
#
# Usage: ./tools/scripts/run_build.sh [zig build args...]
#   e.g. ./tools/scripts/run_build.sh test --summary all
#        ./tools/scripts/run_build.sh lint
#        ./tools/scripts/run_build.sh full-check
#
# Flags:
#   --verbose      Show full environment diagnostics on failure
#   --self-test    Validate Darwin pipeline components without building

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/zig_toolchain.sh"

cd "$(abi_toolchain_repo_root)"

VERBOSE=0
SELF_TEST=0
BUILD_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --verbose) VERBOSE=1 ;;
        --self-test) SELF_TEST=1 ;;
        *) BUILD_ARGS+=("$arg") ;;
    esac
done

ZIG="$(abi_toolchain_resolve_active_zig)"
SYSROOT="${SDKROOT:-$(xcrun --show-sdk-path 2>/dev/null || echo /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk)}"
# Platform version: use live host version for the build runner (a host tool).
# This differs from build/link.zig which uses clamped 15.0 for target artifacts.
MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo 26.0)"

REPO_ROOT="$(pwd)"
RT_CACHE="$REPO_ROOT/.zig-cache/.compiler_rt_path"
ZIGVERSION_FILE="$REPO_ROOT/.zigversion"

# ── Helpers ────────────────────────────────────────────────────────────────

log() { echo "[run_build] $*" >&2; }

print_env() {
    log "Environment:"
    log "  zig:      $ZIG"
    log "  version:  $("$ZIG" version 2>/dev/null || echo unknown)"
    log "  expected: $(cat "$ZIGVERSION_FILE" 2>/dev/null || echo unknown)"
    log "  sdk:      $SYSROOT"
    log "  macOS:    $MACOS_VER"
    log "  ld:       $(command -v /usr/bin/ld 2>/dev/null && /usr/bin/ld -version_detail 2>/dev/null | head -1 || echo 'not found')"
}

# Find compiler_rt from stderr, cache, or filesystem walk
find_compiler_rt() {
    local from_stderr=""

    # 1. Try stderr (only available during build failure path)
    if [[ -n "${STDERR_FILE:-}" && -f "${STDERR_FILE:-}" ]]; then
        from_stderr="$(grep -oE '/[^ )]*libcompiler_rt\.a' "$STDERR_FILE" | head -1 || true)"
        if [[ -n "$from_stderr" && -f "$from_stderr" ]]; then
            # Cache for next time
            echo "$from_stderr" > "$RT_CACHE" 2>/dev/null || true
            echo "$from_stderr"
            return
        fi
    fi

    # 2. Try cache (invalidate if .zigversion is newer)
    if [[ -f "$RT_CACHE" ]]; then
        if [[ -f "$ZIGVERSION_FILE" && "$ZIGVERSION_FILE" -nt "$RT_CACHE" ]]; then
            rm -f "$RT_CACHE"
        else
            local cached
            cached="$(cat "$RT_CACHE" 2>/dev/null || true)"
            if [[ -n "$cached" && -f "$cached" ]]; then
                echo "$cached"
                return
            fi
            rm -f "$RT_CACHE"
        fi
    fi

    # 3. Walk global cache
    local found
    found="$(find "${HOME}/.cache/zig/o" -name 'libcompiler_rt.a' -print -quit 2>/dev/null || true)"
    if [[ -n "$found" && -f "$found" ]]; then
        echo "$found" > "$RT_CACHE" 2>/dev/null || true
        echo "$found"
    fi
}

diagnose_failure() {
    local stderr_file="$1"

    # Check for common failure modes and provide targeted guidance
    if ! [[ -d "$SYSROOT" ]]; then
        log "HINT: SDK not found at $SYSROOT"
        log "  Install Xcode or Command Line Tools: xcode-select --install"
    fi

    local expected_ver
    expected_ver="$(cat "$ZIGVERSION_FILE" 2>/dev/null || true)"
    if [[ -n "$expected_ver" ]]; then
        local actual_ver
        actual_ver="$("$ZIG" version 2>/dev/null || true)"
        if [[ -n "$actual_ver" && "$actual_ver" != "$expected_ver" ]]; then
            log "HINT: Zig version mismatch — expected $expected_ver, got $actual_ver"
        fi
    fi

    if [[ $VERBOSE -eq 1 ]]; then
        print_env
        log "Full stderr:"
        cat "$stderr_file" >&2
    else
        log "Original error (last 20 lines):"
        tail -20 "$stderr_file" >&2
        log "(use --verbose for full output)"
    fi
}

# ── Self-test mode ─────────────────────────────────────────────────────────

run_self_test() {
    local pass=0 fail=0

    check() {
        local name="$1" result="$2" detail="${3:-}"
        if [[ "$result" == "ok" ]]; then
            echo "  PASS  $name${detail:+ ($detail)}"
            pass=$((pass + 1))
        else
            echo "  FAIL  $name${detail:+ ($detail)}"
            fail=$((fail + 1))
        fi
    }

    echo "Darwin pipeline self-test"
    echo "========================="

    # Zig binary
    if command -v "$ZIG" &>/dev/null; then
        local ver
        ver="$("$ZIG" version 2>/dev/null || echo unknown)"
        local expected
        expected="$(cat "$ZIGVERSION_FILE" 2>/dev/null || echo unknown)"
        if [[ "$ver" == "$expected" ]]; then
            check "zig binary" "ok" "$ver"
        else
            check "zig binary" "ok" "found $ver (expected $expected — version mismatch)"
        fi
    else
        check "zig binary" "fail" "not found"
    fi

    # SDK
    if [[ -d "$SYSROOT" ]]; then
        check "SDK path" "ok" "$SYSROOT"
    else
        check "SDK path" "fail" "$SYSROOT does not exist"
    fi

    # Frameworks (spot check)
    if [[ -d "$SYSROOT/System/Library/Frameworks/CoreFoundation.framework" ]]; then
        check "CoreFoundation.framework" "ok"
    else
        check "CoreFoundation.framework" "fail" "not in SDK"
    fi

    # Apple ld
    if [[ -x /usr/bin/ld ]]; then
        local ld_ver
        ld_ver="$(/usr/bin/ld -version_detail 2>/dev/null | head -1 || echo 'unknown version')"
        check "Apple ld" "ok" "$ld_ver"
    else
        check "Apple ld" "fail" "/usr/bin/ld not found or not executable"
    fi

    # compiler_rt
    local rt
    rt="$(find_compiler_rt)"
    if [[ -n "$rt" ]]; then
        check "compiler_rt" "ok" "$rt"
    else
        check "compiler_rt" "fail" "not found — run zig build on a supported platform first"
    fi

    # Compile + relink a trivial program
    local tmpdir
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' RETURN

    cat > "$tmpdir/test.zig" <<'ZIGEOF'
pub export fn main() callconv(.c) c_int {
    return 0;
}
ZIGEOF

    local compile_ok=0
    if "$ZIG" build-obj -femit-bin="$tmpdir/test.o" "$tmpdir/test.zig" 2>/dev/null; then
        compile_ok=1
        check "zig compile" "ok" "trivial program compiled"
    else
        check "zig compile" "fail" "could not compile trivial program"
    fi

    if [[ $compile_ok -eq 1 ]]; then
        if /usr/bin/ld -dynamic \
            -platform_version macos "$MACOS_VER" "$MACOS_VER" \
            -syslibroot "$SYSROOT" \
            -e _main \
            -o "$tmpdir/test" \
            "$tmpdir/test.o" \
            -lSystem \
            ${rt:+"$rt"} 2>/dev/null; then
            check "Apple ld relink" "ok" "trivial program linked"

            if "$tmpdir/test" 2>/dev/null; then
                check "execution" "ok" "trivial program ran successfully"
            else
                check "execution" "fail" "linked binary exited with error"
            fi
        else
            check "Apple ld relink" "fail" "could not link trivial program"
        fi
    fi

    echo ""
    echo "Results: $pass passed, $fail failed"
    [[ $fail -eq 0 ]] && return 0 || return 1
}

if [[ $SELF_TEST -eq 1 ]]; then
    run_self_test
    exit $?
fi

# ── Step 1: Try zig build normally ───────────────────────────────────────

STDERR_FILE="$(mktemp)"
trap 'rm -f "$STDERR_FILE"' EXIT

if "$ZIG" build "${BUILD_ARGS[@]}" 2>"$STDERR_FILE"; then
    exit 0
fi

# ── Step 2: Find build_zcu.o ─────────────────────────────────────────────

BUILD_O="$(grep -oE '\.zig-cache/o/[a-f0-9]+/build_zcu\.o' "$STDERR_FILE" | head -1 || true)"
if [[ -z "$BUILD_O" ]]; then
    BUILD_O="$(find .zig-cache/o -name 'build_zcu.o' -newer "$STDERR_FILE" -print -quit 2>/dev/null || true)"
fi

if [[ -z "$BUILD_O" || ! -f "$BUILD_O" ]]; then
    log "ERROR: Not a linker failure (no build_zcu.o found)."
    diagnose_failure "$STDERR_FILE"
    exit 1
fi

BUILD_DIR="$(dirname "$BUILD_O")"
BUILD_BIN="$BUILD_DIR/build"

# ── Step 3: Find compiler_rt ─────────────────────────────────────────────

COMPILER_RT="$(find_compiler_rt)"
RT_ARGS=()
if [[ -n "$COMPILER_RT" ]]; then
    RT_ARGS=("$COMPILER_RT")
fi

log "Relinking build runner with Apple ld..."
log "  obj: $BUILD_O"
log "  sdk: $SYSROOT"

# ── Step 4: Link with Apple's ld ─────────────────────────────────────────

LINK_START="$(date +%s)"

/usr/bin/ld -dynamic \
    -platform_version macos "$MACOS_VER" "$MACOS_VER" \
    -syslibroot "$SYSROOT" \
    -e _main \
    -o "$BUILD_BIN" \
    "$BUILD_O" \
    -lSystem \
    "${RT_ARGS[@]}" || {
    log "ERROR: Apple ld also failed."
    if [[ $VERBOSE -eq 1 ]]; then print_env; fi
    exit 1
}

LINK_END="$(date +%s)"
LINK_SECS=$(( LINK_END - LINK_START ))
if [[ $LINK_SECS -gt 0 ]]; then
    log "Relinked in ${LINK_SECS}s"
fi

# ── Step 5: Execute the build runner ─────────────────────────────────────

ZIG_LIB_DIR="$("$ZIG" env 2>/dev/null | grep '\.lib_dir' | sed 's/.*= *"\(.*\)".*/\1/' || true)"
if [[ -z "$ZIG_LIB_DIR" ]]; then
    ZIG_LIB_DIR="$(dirname "$(dirname "$ZIG")")/lib"
fi

log "Executing build runner..."
exec "$BUILD_BIN" "$ZIG" "$ZIG_LIB_DIR" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" "${BUILD_ARGS[@]}"
