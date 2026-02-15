#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

errors=0

scan_forbidden() {
    local pattern="$1"
    local label="$2"
    if rg -n --glob '*.zig' "$pattern" src build tools >/tmp/abi_zig016_scan.out 2>/dev/null; then
        echo "ERROR: Found forbidden Zig 0.16 pattern: $label"
        cat /tmp/abi_zig016_scan.out
        errors=$((errors + 1))
    fi
}

# Removed or deprecated Zig 0.15-era APIs/patterns.
scan_forbidden '^[[:space:]]*[^/].*std\.fs\.cwd\(' 'std.fs.cwd() -> use std.Io.Dir.cwd()'
scan_forbidden '^[[:space:]]*[^/].*std\.io\.fixedBufferStream\(' 'std.io.fixedBufferStream() removed in Zig 0.16'
scan_forbidden '^[[:space:]]*[^/].*std\.time\.nanoTimestamp\(' 'std.time.nanoTimestamp() removed in Zig 0.16'
scan_forbidden '^[[:space:]]*[^/].*std\.time\.sleep\(' 'std.time.sleep() forbidden; use services/shared/time wrapper'
scan_forbidden '^[[:space:]]*[^/].*std\.process\.getEnvVar\(' 'std.process.getEnvVar() removed in Zig 0.16'
scan_forbidden '^[[:space:]]*[^/].*@typeInfo\([^)]*\)\.Fn' '@typeInfo(.Fn) -> @typeInfo(.@"fn")'
scan_forbidden '^[[:space:]]*[^/].*std\.ArrayList\([^)]*\)\.init\(' 'std.ArrayList(T).init() legacy usage; prefer ArrayListUnmanaged patterns'

# Formatting-time enum/error name conversion should use {t}, not @tagName/@errorName.
if rg -n --glob '*.zig' 'std\.(debug\.print|log\.[a-z]+)\([^)]*@tagName\(' src build tools >/tmp/abi_zig016_fmt.out 2>/dev/null; then
    echo "ERROR: @tagName() used in print/log formatting context; use {t} instead"
    cat /tmp/abi_zig016_fmt.out
    errors=$((errors + 1))
fi
if rg -n --glob '*.zig' 'std\.(debug\.print|log\.[a-z]+)\([^)]*@errorName\(' src build tools >/tmp/abi_zig016_fmt_err.out 2>/dev/null; then
    echo "ERROR: @errorName() used in print/log formatting context; use {t} instead"
    cat /tmp/abi_zig016_fmt_err.out
    errors=$((errors + 1))
fi

# Test discovery should use `test { _ = @import(...); }`, not comptime blocks.
if rg -n --glob '*.zig' '^[[:space:]]*[^/].*comptime[[:space:]]*\{[[:space:]]*_[[:space:]]*=[[:space:]]*@import\(' src >/tmp/abi_zig016_tests.out 2>/dev/null; then
    echo "ERROR: legacy comptime-based test discovery detected; use test { _ = @import(...); }"
    cat /tmp/abi_zig016_tests.out
    errors=$((errors + 1))
fi

rm -f /tmp/abi_zig016_scan.out /tmp/abi_zig016_fmt.out /tmp/abi_zig016_fmt_err.out /tmp/abi_zig016_tests.out

if [[ "$errors" -gt 0 ]]; then
    echo "FAILED: Zig 0.16 pattern check found $errors issue(s)"
    exit 1
fi

echo "OK: Zig 0.16 pattern checks passed"
