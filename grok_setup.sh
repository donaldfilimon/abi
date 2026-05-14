#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ABI_GROK_API_KEY:-}" ]]; then
  cat >&2 <<'MSG'
Set ABI_GROK_API_KEY in your shell or password manager before running Grok connector tests.
This helper intentionally does not contain a key.
MSG
  exit 1
fi

export ABI_GROK_API_KEY
