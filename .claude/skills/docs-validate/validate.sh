#!/usr/bin/env bash
# Thin wrapper — canonical driver is .agents/skills/docs-validate/validate.sh
# (sync-clis does not copy .sh launchers).
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "$ROOT/.agents/skills/docs-validate/validate.sh" "$@"
