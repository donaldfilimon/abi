#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -f scripts/project_baseline.env ]]; then
    echo "ERROR: scripts/project_baseline.env not found"
    exit 1
fi

# shellcheck disable=SC1091
source scripts/project_baseline.env

errors=0

zig_version_file="$(tr -d '[:space:]' < .zigversion)"
if [[ "$zig_version_file" != "$ABI_ZIG_VERSION" ]]; then
    echo "ERROR: .zigversion ($zig_version_file) does not match baseline ($ABI_ZIG_VERSION)"
    exit 1
fi

if ! command -v zig >/dev/null 2>&1; then
    echo "ERROR: no 'zig' binary found on PATH"
    exit 1
fi

active_zig="$(command -v zig)"
active_zig_version="$(zig version | tr -d '[:space:]')"
if [[ "$active_zig_version" != "$ABI_ZIG_VERSION" ]]; then
    echo "ERROR: active zig version ($active_zig_version from $active_zig) does not match pinned baseline ($ABI_ZIG_VERSION)"
    errors=$((errors + 1))
fi

# When zvm is installed, enforce deterministic PATH precedence through .zvm/bin.
if command -v zvm >/dev/null 2>&1; then
    zvm_zig="$HOME/.zvm/bin/zig"
    if [[ -x "$zvm_zig" && "$active_zig" != "$zvm_zig" ]]; then
        echo "ERROR: PATH precedence mismatch: active zig is '$active_zig' but zvm-managed zig is '$zvm_zig'"
        echo "       Fix by prepending '$HOME/.zvm/bin' ahead of other zig locations in PATH."
        errors=$((errors + 1))
    fi
fi

declare -a files=(
    "README.md"
    "CONTRIBUTING.md"
    "CLAUDE.md"
    "AGENTS.md"
    "docs/README.md"
    "docs/roadmap.md"
    "docs/content/getting-started.md"
)

version_regex='0\.16\.0-dev\.[0-9]+\+[A-Za-z0-9]+'

for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: expected file missing: $file"
        errors=$((errors + 1))
        continue
    fi

    if ! grep -q "$ABI_ZIG_VERSION" "$file"; then
        echo "ERROR: $file does not mention expected Zig version: $ABI_ZIG_VERSION"
        errors=$((errors + 1))
    fi

    while IFS=: read -r lineno match; do
        if [[ "$match" != "$ABI_ZIG_VERSION" ]]; then
            echo "ERROR: $file:$lineno has mismatched Zig version '$match' (expected '$ABI_ZIG_VERSION')"
            errors=$((errors + 1))
        fi
    done < <(grep -Eno "$version_regex" "$file" || true)
done

if [[ "$errors" -gt 0 ]]; then
    echo "FAILED: Zig version consistency check found $errors issue(s)"
    echo "Hint: run 'bash scripts/toolchain_doctor.sh' for a full local diagnosis."
    exit 1
fi

echo "OK: Zig version consistency checks passed"
