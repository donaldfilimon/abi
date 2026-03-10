#!/usr/bin/env bash
# .cel/lib.sh — Shared shell helpers for CEL toolchain scripts.
#
# Source this from build.sh, cel_migrate.sh, and use_cel.sh to avoid
# duplicating stock-zig detection, version probing, and build-runner
# classification logic.
#
# Requires: CEL_DIR and REPO_ROOT to be set before sourcing.

# ── Logging ────────────────────────────────────────────────────────────
cel_info()  { printf "\033[1;34m[cel]\033[0m %s\n" "$*"; }
cel_warn()  { printf "\033[1;33m[cel]\033[0m %s\n" "$*"; }
cel_error() { printf "\033[1;31m[cel]\033[0m %s\n" "$*" >&2; }
cel_ok()    { printf "\033[0;32m[cel]\033[0m %s\n" "$*"; }
cel_die()   { cel_error "$@"; exit 1; }

# ── Binary name (Windows compat) ──────────────────────────────────────
cel_binary_name() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) printf '%s.exe' "$1" ;;
        *) printf '%s' "$1" ;;
    esac
}

# ── Stock Zig detection ───────────────────────────────────────────────
cel_stock_zig_path() {
    if command -v zig >/dev/null 2>&1; then
        command -v zig
    else
        return 1
    fi
}

cel_stock_zig_version() {
    local path
    path="$(cel_stock_zig_path)" || return 1
    "$path" version 2>/dev/null | tr -d '\r' | head -n 1
}

# ── Expected version from .zigversion ─────────────────────────────────
cel_expected_zig_version() {
    if [[ -n "${ZIG_VERSION:-}" ]]; then
        printf '%s' "$ZIG_VERSION"
    elif [[ -f "${REPO_ROOT:-.}/.zigversion" ]]; then
        tr -d '[:space:]' < "${REPO_ROOT:-.}/.zigversion"
    else
        return 1
    fi
}

# ── Build-runner classification ───────────────────────────────────────
# Returns one of: ok, cel-active, darwin-linker, failing, missing
cel_classify_build_runner() {
    local cel_bin="${CEL_DIR:-${REPO_ROOT:-.}/.cel}/bin/$(cel_binary_name zig)"
    local path

    path="$(cel_stock_zig_path)" || {
        printf 'missing'
        return 0
    }

    if [[ "$path" == "$cel_bin" ]]; then
        printf 'cel-active'
        return 0
    fi

    local output
    if output="$(cd "${REPO_ROOT:-.}" && zig build --help 2>&1 1>/dev/null)"; then
        printf 'ok'
        return 0
    fi

    if [[ "$output" == *"__availability_version_check"* || "$output" == *"undefined symbol:"* ]]; then
        printf 'darwin-linker'
        return 0
    fi

    printf 'failing'
}

# ── Platform detection ────────────────────────────────────────────────
cel_is_blocked_darwin() {
    [[ "$(uname -s)" == "Darwin" ]] || return 1
    local os_ver major
    os_ver="$(sw_vers -productVersion 2>/dev/null || echo 0)"
    major="${os_ver%%.*}"
    [[ "$major" =~ ^[0-9]+$ ]] || return 1
    (( major >= ${CEL_MIN_MACOS_MAJOR:-26} ))
}

# ── Parallel job count ────────────────────────────────────────────────
cel_job_count() {
    local jobs
    jobs="${CMAKE_JOBS:-$(( ($(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) + 1) / 2 ))}"
    [[ "$jobs" -lt 1 ]] && jobs=1
    printf '%s' "$jobs"
}
