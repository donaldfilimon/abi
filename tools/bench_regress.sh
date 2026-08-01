#!/usr/bin/env bash
# Local Rust benchmark regression gate for the frozen WDBX CLI workload.
#
# This is a same-system/architecture development guard, not a published
# throughput claim. The debug benchmark is intentionally aggregated as the
# best p50 of several runs so transient host load only makes the gate more
# conservative; cross-host comparisons are refused by default.
#
# Usage:
#   ./tools/bench_regress.sh
#   ./tools/bench_regress.sh --update-baseline
#
# Environment:
#   ABI_BENCH_COUNT              vectors per run (default: 500)
#   ABI_BENCH_RUNS               suite runs, best p50 kept (default: 5)
#   ABI_BENCH_THRESHOLD_PCT      allowed slowdown percent (default: 25)
#   ABI_BENCH_ALLOW_CROSS_HOST   set to 1 to compare unlike OS/architectures
#
# Deterministic test hook (both values are required together):
#   ABI_BENCH_CURRENT_INSERT_P50_NS
#   ABI_BENCH_CURRENT_SEARCH_P50_NS
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ABI="$ROOT/target/debug/abi"
BASELINE="$ROOT/tools/bench_baseline.json"
COUNT="${ABI_BENCH_COUNT:-500}"
RUNS="${ABI_BENCH_RUNS:-5}"
THRESHOLD_PCT="${ABI_BENCH_THRESHOLD_PCT:-25}"
SYSTEM="$(uname -s)"
MACHINE="$(uname -m)"
UPDATE_BASELINE=0

usage() {
  sed -n '2,/^set -euo pipefail$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
}

case "${1:-}" in
  "") ;;
  --update-baseline) UPDATE_BASELINE=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
  echo "error: too many arguments (try --help)" >&2
  exit 2
fi

for value_name in COUNT RUNS THRESHOLD_PCT; do
  value="${!value_name}"
  case "$value" in
    ""|*[!0-9]*)
      echo "error: $value_name must be a non-negative integer (got '$value')" >&2
      exit 2
      ;;
  esac
done
if [ "$COUNT" -lt 1 ] || [ "$RUNS" -lt 1 ]; then
  echo "error: ABI_BENCH_COUNT and ABI_BENCH_RUNS must be at least 1" >&2
  exit 2
fi

parse_metrics() {
  awk '
    /^[[:space:]]+p50=/ {
      seen += 1
      value = $1
      sub(/^p50=/, "", value)
      if (seen == 1) insert = value
      if (seen == 2) {
        print insert " " value
        exit
      }
    }
  '
}

run_suite() {
  if [ ! -x "$ABI" ]; then
    echo "error: $ABI is missing; run ./tools/cargo.sh build -p abi-cli" >&2
    exit 1
  fi

  best_insert=""
  best_search=""
  run=1
  while [ "$run" -le "$RUNS" ]; do
    output="$($ABI wdbx benchmark "$COUNT" 2>&1)"
    metrics="$(printf '%s\n' "$output" | parse_metrics)"
    insert="${metrics%% *}"
    search="${metrics##* }"
    case "$insert:$search" in
      *[!0-9:]*|":"|*":")
        echo "error: could not parse benchmark p50 values from run $run" >&2
        printf '%s\n' "$output" >&2
        exit 1
        ;;
    esac
    printf 'bench run %d/%d: insert_p50=%s ns search_p50=%s ns\n' \
      "$run" "$RUNS" "$insert" "$search"
    if [ -z "$best_insert" ] || [ "$insert" -lt "$best_insert" ]; then
      best_insert="$insert"
    fi
    if [ -z "$best_search" ] || [ "$search" -lt "$best_search" ]; then
      best_search="$search"
    fi
    run=$((run + 1))
  done
  CURRENT_INSERT="$best_insert"
  CURRENT_SEARCH="$best_search"
}

hook_insert="${ABI_BENCH_CURRENT_INSERT_P50_NS:-}"
hook_search="${ABI_BENCH_CURRENT_SEARCH_P50_NS:-}"
if [ -n "$hook_insert" ] || [ -n "$hook_search" ]; then
  if [ -z "$hook_insert" ] || [ -z "$hook_search" ]; then
    echo "error: both deterministic benchmark hook values must be set together" >&2
    exit 2
  fi
  case "$hook_insert:$hook_search" in
    *[!0-9:]*|":"|*":")
      echo "error: deterministic benchmark hook values must be positive integers" >&2
      exit 2
      ;;
  esac
  if [ "$hook_insert" -lt 1 ] || [ "$hook_search" -lt 1 ]; then
    echo "error: deterministic benchmark hook values must be positive integers" >&2
    exit 2
  fi
  if [ "$UPDATE_BASELINE" -eq 1 ]; then
    echo "error: refusing to update the baseline from deterministic test-hook values" >&2
    exit 2
  fi
  CURRENT_INSERT="$hook_insert"
  CURRENT_SEARCH="$hook_search"
  echo "bench: using deterministic comparison values"
else
  run_suite
fi

if [ "$UPDATE_BASELINE" -eq 1 ]; then
  commit="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || printf unknown)"
  cat > "$BASELINE" <<EOF
{
  "schema": "abi-rust-bench-baseline/v1",
  "workload": "abi wdbx benchmark",
  "count": $COUNT,
  "runs": $RUNS,
  "aggregation": "best p50_ns",
  "system": "$SYSTEM",
  "machine": "$MACHINE",
  "commit": "$commit",
  "measurements": {
    "insert_p50_ns": $CURRENT_INSERT,
    "search_p50_ns": $CURRENT_SEARCH
  }
}
EOF
  echo "baseline updated: $BASELINE"
  exit 0
fi

if [ ! -f "$BASELINE" ]; then
  echo "error: missing benchmark baseline at $BASELINE" >&2
  echo "       Generate it with ./tools/bench_regress.sh --update-baseline" >&2
  exit 1
fi

json_string() {
  key="$1"
  sed -n "s/^[[:space:]]*\"$key\": \"\([^\"]*\)\".*/\1/p" "$BASELINE"
}
json_integer() {
  key="$1"
  sed -n "s/^[[:space:]]*\"$key\": \([0-9][0-9]*\).*/\1/p" "$BASELINE"
}

schema="$(json_string schema)"
base_system="$(json_string system)"
base_machine="$(json_string machine)"
base_count="$(json_integer count)"
base_runs="$(json_integer runs)"
base_insert="$(json_integer insert_p50_ns)"
base_search="$(json_integer search_p50_ns)"

if [ "$schema" != "abi-rust-bench-baseline/v1" ]; then
  echo "error: unsupported benchmark baseline schema '$schema'" >&2
  exit 1
fi
if [ "$base_count" != "$COUNT" ] || [ "$base_runs" != "$RUNS" ]; then
  echo "error: benchmark workload mismatch: baseline count/runs=$base_count/$base_runs, current=$COUNT/$RUNS" >&2
  echo "       Use the baseline workload or refresh it deliberately." >&2
  exit 1
fi
if [ "$base_system" != "$SYSTEM" ] || [ "$base_machine" != "$MACHINE" ]; then
  if [ "${ABI_BENCH_ALLOW_CROSS_HOST:-0}" != "1" ]; then
    echo "benchmark regression gate: SKIP (no comparable baseline for $SYSTEM/$MACHINE; committed baseline is $base_system/$base_machine)"
    exit 0
  fi
  echo "warning: comparing benchmark timings across host classes" >&2
fi
if [ -z "$base_insert" ] || [ -z "$base_search" ] || \
   [ "$base_insert" -lt 1 ] || [ "$base_search" -lt 1 ]; then
  echo "error: benchmark baseline measurements are missing or invalid" >&2
  exit 1
fi

failed=0
compare() {
  label="$1"
  baseline="$2"
  current="$3"
  allowed=$((baseline + (baseline * THRESHOLD_PCT + 99) / 100))
  if [ "$current" -gt "$allowed" ]; then
    echo "REGRESS $label: baseline=$baseline ns current=$current ns allowed=$allowed ns (+$THRESHOLD_PCT%)"
    failed=$((failed + 1))
  else
    echo "OK      $label: baseline=$baseline ns current=$current ns allowed=$allowed ns (+$THRESHOLD_PCT%)"
  fi
}

echo "bench: best p50 of $RUNS run(s), count=$COUNT, host=$SYSTEM/$MACHINE"
compare "HNSW insert" "$base_insert" "$CURRENT_INSERT"
compare "HNSW search" "$base_search" "$CURRENT_SEARCH"
if [ "$failed" -ne 0 ]; then
  echo "benchmark regression gate: FAIL ($failed measurement(s))" >&2
  exit 1
fi
echo "benchmark regression gate: PASS"
