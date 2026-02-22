#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_cli_full_matrix.py"
STUB_ENV="${SCRIPT_DIR}/cli_full_env.stub"

if [[ ! -f "$RUNNER" ]]; then
  echo "runner missing: $RUNNER" >&2
  exit 1
fi

if [[ ! -f "$STUB_ENV" ]]; then
  echo "stub env missing: $STUB_ENV" >&2
  exit 1
fi

TMP_ENV="$(mktemp)"
trap 'rm -f "$TMP_ENV"' EXIT
cp "$STUB_ENV" "$TMP_ENV"

ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO_ROOT="$2"
      shift 2
      ;;
    --env-file)
      cp "$2" "$TMP_ENV"
      shift 2
      ;;
    -h|--help)
      python3 "$RUNNER" --help
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

python3 "$RUNNER" --repo "$REPO_ROOT" --allow-blocked --env-file "$TMP_ENV" "${ARGS[@]}"
