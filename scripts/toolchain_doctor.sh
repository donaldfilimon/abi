#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

expected_version="$(tr -d '[:space:]' < .zigversion)"

echo "ABI toolchain doctor"
echo "Repo: $repo_root"
echo "Pinned Zig (.zigversion): $expected_version"
echo

if ! command -v zig >/dev/null 2>&1; then
    echo "ERROR: no 'zig' binary found on PATH"
    echo "Install via zvm and ensure ~/.zvm/bin is on PATH."
    exit 1
fi

echo "Active zig:"
active_zig="$(command -v zig)"
active_version="$(zig version | tr -d '[:space:]')"
echo "  path:    $active_zig"
echo "  version: $active_version"
echo

echo "All zig candidates on PATH (in precedence order):"
if command -v which >/dev/null 2>&1; then
    which -a zig 2>/dev/null | awk '!seen[$0]++ { print "  - " $0 }'
else
    echo "  - (which unavailable; skipped)"
fi
echo

issues=0

if [[ "$active_version" != "$expected_version" ]]; then
    echo "ISSUE: active zig version does not match .zigversion"
    issues=$((issues + 1))
fi

zvm_zig="$HOME/.zvm/bin/zig"
if [[ -x "$zvm_zig" && "$active_zig" != "$zvm_zig" ]]; then
    echo "ISSUE: active zig is not the zvm-managed binary"
    issues=$((issues + 1))
fi

if (( issues == 0 )); then
    echo "OK: local Zig toolchain is deterministic and matches repository pin."
    exit 0
fi

echo
echo "Suggested fix:"
if command -v zvm >/dev/null 2>&1; then
    echo "  1) zvm upgrade"
    echo "  2) zvm install \"$expected_version\""
    echo "  3) zvm use \"$expected_version\""
    echo "  4) export PATH=\"\$HOME/.zvm/bin:\$PATH\""
    echo "  5) hash -r"
    echo "  6) bash scripts/check_zig_version_consistency.sh"
else
    echo "  1) Install Zig $expected_version from https://ziglang.org/download/"
    echo "  2) Put the Zig binary on PATH ahead of other installations"
    echo "  3) hash -r"
    echo "  4) bash scripts/check_zig_version_consistency.sh"
fi
echo
echo "FAILED: toolchain doctor found $issues issue(s)."
exit 1
