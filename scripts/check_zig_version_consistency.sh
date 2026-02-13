#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pinned_version="$(tr -d '[:space:]' < .zigversion)"
minimum_version="$(sed -n 's/^[[:space:]]*\.minimum_zig_version = "\(.*\)",.*/\1/p' build.zig.zon | head -n 1)"

if [[ -z "$pinned_version" ]]; then
    echo "ERROR: .zigversion is empty." >&2
    exit 1
fi

if [[ -z "$minimum_version" ]]; then
    echo "ERROR: Could not read .minimum_zig_version from build.zig.zon." >&2
    exit 1
fi

if [[ "$pinned_version" != "$minimum_version" ]]; then
    echo "ERROR: Zig version mismatch:" >&2
    echo "  .zigversion:           $pinned_version" >&2
    echo "  build.zig.zon minimum: $minimum_version" >&2
    exit 1
fi

has_errors=0
matches_file="$(mktemp)"
trap 'rm -f "$matches_file"' EXIT
badge_version="${pinned_version//-/--}"
badge_version="${badge_version//+/%2B}"

check_pattern() {
    local pattern="$1"
    local description="$2"
    if rg -n --glob '*.md' --glob '*.html' --pcre2 "$pattern" . >"$matches_file"; then
        echo "ERROR: Found ${description}:" >&2
        cat "$matches_file" >&2
        has_errors=1
    fi
}

check_pattern '0\.16\.x' "wildcard Zig version strings (0.16.x)"
check_pattern '0\.16\.0-dev\.2471\+e9eadee00' "stale pinned version references"
check_pattern 'Zig 0\.16($|[^0-9.])' "ambiguous Zig 0.16 references"
check_pattern "img\\.shields\\.io/badge/Zig-(?!${badge_version}\\b)" "non-canonical Zig badge values"

if [[ "$has_errors" -ne 0 ]]; then
    exit 1
fi

echo "OK: Zig version metadata and docs are consistent with $pinned_version."
