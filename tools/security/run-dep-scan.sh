#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

if command -v cargo-audit >/dev/null 2>&1; then
    echo "dependency-scan: RUN cargo-audit ($(cargo-audit --version))"
    echo "dependency-scan: scoped exceptions RUSTSEC-2025-0141,RUSTSEC-2024-0436 (optional pinned TFHE-rs transitives)"
    exec ./tools/cargo.sh audit --deny warnings \
        --ignore RUSTSEC-2025-0141 \
        --ignore RUSTSEC-2024-0436
fi

if command -v cargo-deny >/dev/null 2>&1; then
    echo "dependency-scan: RUN cargo-deny ($(cargo-deny --version))"
    exec ./tools/cargo.sh deny check advisories bans licenses sources
fi

echo "dependency-scan: SKIP — install cargo-audit or cargo-deny; no dependency scanner ran" >&2
echo "dependency-scan: DAST, Semgrep, Guardian, and hosted CodeQL are separate evidence" >&2

if [ "${ABI_DEP_SCAN_REQUIRE:-0}" = "1" ]; then
    echo "dependency-scan: FAIL — ABI_DEP_SCAN_REQUIRE=1" >&2
    exit 2
fi

exit 0
