#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! bash scripts/check_feature_catalog_audit.sh; then
    echo "FAILED: Feature catalog consistency check failed"
    exit 1
fi
echo "OK: Feature catalog consistency check passed"
