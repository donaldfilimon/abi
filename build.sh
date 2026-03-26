#!/bin/bash
set -euo pipefail

# ABI Build Wrapper for macOS 26.4+
# Works around Zig's Mach-O linker incompatibility with macOS 26+ SDK TBD files.
#
# Root cause: macOS 26+ SDK .tbd files list "arm64e-macos" but NOT "arm64-macos".
# Zig's self-hosted linker targets plain arm64 and cannot match arm64e targets,
# causing all system symbol resolution to fail.
#
# Fix: Create a patched SDK overlay with arm64-macos added to TBD target lists,
# then pass --sysroot pointing to the overlay. The build runner is also relinked
# with Apple's native /usr/bin/ld since it links BEFORE build.zig runs.
#
# Usage: ./build.sh [--link] [zig build args...]
#        ./build.sh std                — run zig std (std library check)
#        ./build.sh test               — run zig test src/root.zig directly
#        ./build.sh run <file.zig>     — run zig run <file.zig>
# Example: ./build.sh test --summary all
#          ./build.sh --link lib
#          ./build.sh -Dfeat-gpu=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUTO_LINK=false

# Check for --link flag
ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--link" ]; then
        AUTO_LINK=true
    else
        ARGS="$ARGS $arg"
    fi
done
set -- $ARGS

# Resolve zig via zigup.sh (auto-downloads if missing)
ZIG2="$("$SCRIPT_DIR/tools/zigup.sh" --status)"
ZIG_LIB="$(dirname "$(dirname "$ZIG2")")/lib"

SYSROOT="$(xcrun --show-sdk-path 2>/dev/null || echo "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")"
MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo "26.4")"
DARWIN_MAJOR="$(uname -r | cut -d. -f1)"

STDERR_FILE="$(mktemp)"
TEST_STDERR="$(mktemp)"
trap 'rm -f "$STDERR_FILE" "$TEST_STDERR"' EXIT

# ── SDK overlay for Darwin 25+ ──────────────────────────────────────
# Create patched SDK overlay if on macOS 26+ (Darwin 25+)
SDK_OVERLAY=""
SYSROOT_ARGS=""
if [ "$(uname -s)" = "Darwin" ] && [ "$DARWIN_MAJOR" -ge 25 ] 2>/dev/null; then
    SDK_OVERLAY="$("$SCRIPT_DIR/tools/patch-sdk.sh" 2>/dev/null || true)"
    if [ -n "$SDK_OVERLAY" ] && [ -d "$SDK_OVERLAY" ]; then
        SYSROOT_ARGS="--sysroot $SDK_OVERLAY"
        echo "[darwin-wrapper] Using patched SDK overlay: $SDK_OVERLAY" >&2
    fi
fi

# ── Helper functions ────────────────────────────────────────────────

build_runner_arch() {
    local build_bin="$1"
    local archs
    local file_out

    archs="$(lipo -archs "$build_bin" 2>/dev/null || true)"
    if [[ -n "$archs" ]]; then
        set -- $archs
        printf '%s\n' "$1"
        return 0
    fi

    file_out="$(file -b "$build_bin" 2>/dev/null || true)"
    case "$file_out" in
        *arm64*|*aarch64*) printf 'arm64\n'; return 0 ;;
        *x86_64*) printf 'x86_64\n'; return 0 ;;
    esac

    local build_o="${build_bin}_zcu.o"
    if [[ -f "$build_o" ]]; then
        file_out="$(file -b "$build_o" 2>/dev/null || true)"
        case "$file_out" in
            *arm64*|*aarch64*) printf 'arm64\n'; return 0 ;;
            *x86_64*) printf 'x86_64\n'; return 0 ;;
        esac
    fi

    return 1
}

archive_matches_arch() {
    local archive="$1"
    local expected_arch="$2"
    local info arch_list arch

    info="$(lipo -info "$archive" 2>/dev/null || true)"
    [[ -z "$info" ]] && return 1

    case "$info" in
        *" are: "*) arch_list="${info##*: }" ;;
        *" architecture: "*) arch_list="${info##*: }" ;;
        *) return 1 ;;
    esac

    for arch in $arch_list; do
        if [[ "$arch" == "$expected_arch" ]]; then
            return 0
        fi
    done
    return 1
}

find_host_compiler_rt() {
    local expected_arch="$1"
    local archive

    while IFS= read -r archive; do
        [[ -z "$archive" ]] && continue
        if archive_matches_arch "$archive" "$expected_arch"; then
            printf '%s\n' "$archive"
            return 0
        fi
    done < <(ls -t "$HOME/.cache/zig/o/"*/libcompiler_rt.a 2>/dev/null || true)

    return 1
}

run_build() {
    "$ZIG2" build \
        --zig-lib-dir "$ZIG_LIB" \
        --global-cache-dir "$HOME/.cache/zig" \
        --cache-dir .zig-cache \
        $SYSROOT_ARGS \
        "$@" \
        2>"$STDERR_FILE" && return 0
    return 1
}

# ── Direct zig subcommand handler (std, test, run) ──────────────────
# These bypass build.zig entirely and need the same LLD workaround:
# compile the object, then relink with Apple's /usr/bin/ld.

relink_and_run() {
    local obj_file="$1"
    local bin_file="$2"
    local run_args=("${@:3}")
    local obj_arch

    obj_arch="$(file -b "$obj_file" 2>/dev/null || true)"
    local bin_arch=""
    case "$obj_arch" in
        *arm64*|*aarch64*) bin_arch="arm64" ;;
        *x86_64*) bin_arch="x86_64" ;;
    esac

    if [[ -z "$bin_arch" ]]; then
        echo "[darwin-wrapper] Could not determine architecture from $obj_file" >&2
        return 1
    fi

    local crt
    crt="$(find_host_compiler_rt "$bin_arch" || true)"
    if [[ -z "$crt" || ! -f "$crt" ]]; then
        echo "[darwin-wrapper] No host-compatible libcompiler_rt.a found for $bin_arch" >&2
        return 1
    fi

    echo "[darwin-wrapper] Relinking $bin_file with Apple ld ($bin_arch)..." >&2
    /usr/bin/ld -dynamic -platform_version macos "$MACOS_VER" "$MACOS_VER" \
        -syslibroot "$SYSROOT" -e _main -o "$bin_file" "$obj_file" "$crt" "$SYSROOT/usr/lib/libSystem.B.tbd" "$SYSROOT/usr/lib/libc++.tbd" 2>&1

    if [[ ! -x "$bin_file" ]]; then
        echo "[darwin-wrapper] Relink failed: $bin_file not executable" >&2
        return 1
    fi

    echo "[darwin-wrapper] Relinked. Running $bin_file ..." >&2
    "$bin_file" "${run_args[@]+"${run_args[@]}"}"
    return $?
}

# Extract object file path from zig linker error output (first "referenced by" line)
parse_obj_from_errors() {
    grep "referenced by" "$1" 2>/dev/null | head -1 | sed -E 's/.*referenced by ([^:]+):.*/\1/'
}

direct_zig_cmd() {
    local cmd="$1"
    shift

    # Run zig command — expected to fail at link step
    set +e
    "$ZIG2" "$cmd" \
        --zig-lib-dir "$ZIG_LIB" \
        --global-cache-dir "$HOME/.cache/zig" \
        --cache-dir .zig-cache \
        $SYSROOT_ARGS \
        "$@" \
        2>"$STDERR_FILE"
    local zig_rc=$?
    set -e

    # If it succeeded (unlikely on macOS 26+), we're done
    if [ $zig_rc -eq 0 ]; then
        cat "$STDERR_FILE" >&2
        return 0
    fi

    # Check if failure is due to undefined symbols (expected LLD issue)
    if ! grep -q "undefined symbol:" "$STDERR_FILE" 2>/dev/null; then
        # Not a linker error — report original error
        cat "$STDERR_FILE" >&2
        return $zig_rc
    fi

    # Parse the object file from the error output
    local obj_file
    obj_file="$(parse_obj_from_errors "$STDERR_FILE")"

    if [[ -z "$obj_file" || ! -f "$obj_file" ]]; then
        echo "[darwin-wrapper] Could not find object file from linker errors" >&2
        cat "$STDERR_FILE" >&2
        return 1
    fi

    # Derive the binary path (strip _zcu.o suffix)
    local bin_file="${obj_file%_zcu.o}"

    # Build run arguments: pass zig lib dir for std/test to locate test harness
    local run_args=()
    case "$cmd" in
        std)  run_args+=("$ZIG_LIB") ;; # std test suite needs zig lib dir
        test) run_args+=("$ZIG_LIB") ;; # test runner needs zig lib dir
        run)  ;; # user program, no extra args
    esac

    relink_and_run "$obj_file" "$bin_file" "${run_args[@]+"${run_args[@]}"}"
}

# ── Detect direct zig subcommands ───────────────────────────────────
FIRST_ARG="${1:-}"
case "$FIRST_ARG" in
    std)
        shift
        direct_zig_cmd "std" "$@"
        exit $?
        ;;
    test)
        # Only route to direct zig test when no additional args
        # "./build.sh test" → zig test, "./build.sh test --summary all" → zig build test
        if [ $# -eq 1 ]; then
            shift
            direct_zig_cmd "test" "src/root.zig"
            exit $?
        fi
        ;;
    run)
        # Only route to direct zig run if a file arg follows
        SECOND_ARG="${2:-}"
        if [[ -n "$SECOND_ARG" && "$SECOND_ARG" != -* ]]; then
            shift
            direct_zig_cmd "run" "$@"
            exit $?
        fi
        ;;
esac

# ── Try direct build first ──────────────────────────────────────────
if run_build "$@"; then
    if [ "$AUTO_LINK" = true ]; then
        "$SCRIPT_DIR/tools/zigup.sh" --link 2>&1 || true
    fi
    exit 0
fi

# ── Build runner relink (Darwin 25+ only) ───────────────────────────
# The build runner links BEFORE build.zig runs, so the sysroot/overlay
# doesn't help it. Relink with Apple's native /usr/bin/ld.

BUILD_O="$(ls -t .zig-cache/o/*/build_zcu.o 2>/dev/null | head -1 || true)"

if [[ -n "$BUILD_O" && -f "$BUILD_O" ]]; then
    BUILD_DIR="$(dirname "$BUILD_O")"
    BUILD_BIN="$BUILD_DIR/build"
    BUILD_ARCH="$(build_runner_arch "$BUILD_BIN" || true)"

    if [[ -z "$BUILD_ARCH" ]]; then
        echo "[darwin-wrapper] Could not determine build runner architecture from $BUILD_BIN" >&2
        cat "$STDERR_FILE" >&2
        exit 1
    fi

    CRT="$(find_host_compiler_rt "$BUILD_ARCH" || true)"
    if [[ -z "$CRT" || ! -f "$CRT" ]]; then
        echo "[darwin-wrapper] No host-compatible libcompiler_rt.a found for build runner architecture $BUILD_ARCH" >&2
        cat "$STDERR_FILE" >&2
        exit 1
    fi

    echo "[darwin-wrapper] Relinking $BUILD_BIN with Apple ld + compiler_rt ($BUILD_ARCH)..." >&2

    LD_ARGS=(-dynamic -platform_version macos "$MACOS_VER" "$MACOS_VER" -syslibroot "$SYSROOT" -e _main -o "$BUILD_BIN" "$BUILD_O")
    LD_ARGS+=("$CRT")
    LD_ARGS+=("$SYSROOT/usr/lib/libSystem.B.tbd" "$SYSROOT/usr/lib/libc++.tbd")

    /usr/bin/ld "${LD_ARGS[@]}" 2>&1

    if [[ -x "$BUILD_BIN" ]]; then
        echo "[darwin-wrapper] Build runner relinked. Running build..." >&2
        if "$BUILD_BIN" "$ZIG2" "$ZIG_LIB" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" $SYSROOT_ARGS "$@" 2>"$TEST_STDERR"; then
            # Show build output (zig writes to stderr)
            cat "$TEST_STDERR" >&2
            rm -f "$TEST_STDERR"
            if [ "$AUTO_LINK" = true ]; then
                "$SCRIPT_DIR/tools/zigup.sh" --link 2>&1 || true
            fi
            exit 0
        fi

        # Check if failure is due to Accelerate/vDSP link errors
        if grep -q "undefined symbol:.*vDSP\|undefined symbol:.*vvexpf\|undefined symbol:.*vvsqrtf" "$TEST_STDERR" 2>/dev/null; then
            echo "[darwin-wrapper] LLD cannot resolve Accelerate symbols on macOS 26+." >&2
            echo "[darwin-wrapper] Retrying with -Dfeat-gpu=false ..." >&2
            rm -f "$TEST_STDERR"
            "$BUILD_BIN" "$ZIG2" "$ZIG_LIB" "$(pwd)" ".zig-cache" "${HOME}/.cache/zig" $SYSROOT_ARGS -Dfeat-gpu=false "$@"
            EXIT_CODE=$?
            if [ "$AUTO_LINK" = true ]; then
                "$SCRIPT_DIR/tools/zigup.sh" --link 2>&1 || true
            fi
            exit $EXIT_CODE
        fi

        # Other failure — print captured stderr and exit
        cat "$TEST_STDERR" >&2
        rm -f "$TEST_STDERR"
        exit 1
    else
        echo "[darwin-wrapper] Build runner binary not created" >&2
        exit 1
    fi
fi

echo "[darwin-wrapper] Could not find build_zcu.o" >&2
cat "$STDERR_FILE" >&2
exit 1
