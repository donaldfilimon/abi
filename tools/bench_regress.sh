#!/usr/bin/env bash
# bench_regress.sh — benchmark baseline + regression gate for abi.
#
# Wired into `zig build full-check` (release-readiness gate); also runs
# standalone. It runs
# `zig build benchmarks`, which writes the machine-readable artifact
# zig-out/bench/results.json (schema "abi-bench/v2", see
# src/benchmarks.zig BENCH_ARTIFACT_PATH), then compares per-benchmark
# timings against the committed baseline in tools/bench_baseline.json.
#
# Usage:
#   tools/bench_regress.sh                  # run + compare; exit 1 on regression
#   tools/bench_regress.sh --update-baseline  # run + rewrite baseline from this run
#   tools/bench_regress.sh --compare-only   # skip the run; compare existing artifact
#
# Environment:
#   ZIG                       path to the zig binary. Defaults to the pinned
#                             toolchain ~/.zvm/$(cat .zigversion)/zig when it
#                             exists, else `zig` on PATH (version-checked).
#   ABI_BENCH_THRESHOLD_PCT   allowed slowdown percent before failing (default 5).
#   ABI_BENCH_METRIC          which artifact field to compare: min_ms (default),
#                             p50_ms, or avg_ms. min_ms is the least noisy for
#                             a 10-iteration suite; avg/p50 swing more run-to-run.
#   ABI_BENCH_RUNS            how many times to run the whole suite (default 3).
#                             Per benchmark, the record with the smallest metric
#                             value across runs is kept ("best of N") — this is
#                             what makes results usable on a machine with other
#                             builds running; single runs swing far beyond 5%.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT="$REPO_ROOT/zig-out/bench/results.json"
BASELINE="$REPO_ROOT/tools/bench_baseline.json"
THRESHOLD_PCT="${ABI_BENCH_THRESHOLD_PCT:-5}"
METRIC="${ABI_BENCH_METRIC:-min_ms}"
RUNS="${ABI_BENCH_RUNS:-3}"

UPDATE_BASELINE=0
COMPARE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --update-baseline) UPDATE_BASELINE=1 ;;
    --compare-only) COMPARE_ONLY=1 ;;
    -h|--help)
      awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

case "$METRIC" in
  min_ms|p50_ms|avg_ms|p95_ms|p99_ms|max_ms) ;;
  *)
    echo "error: ABI_BENCH_METRIC must be one of min_ms|p50_ms|avg_ms|p95_ms|p99_ms|max_ms (got '$METRIC')" >&2
    exit 2
    ;;
esac

# --- resolve zig (prefer the pinned toolchain; PATH zig may be a stale brew 0.16) ---
resolve_zig() {
  if [[ -n "${ZIG:-}" ]]; then
    echo "$ZIG"
    return
  fi
  local pin=""
  if [[ -f "$REPO_ROOT/.zigversion" ]]; then
    pin="$(tr -d '[:space:]' < "$REPO_ROOT/.zigversion")"
  fi
  if [[ -n "$pin" && -x "$HOME/.zvm/$pin/zig" ]]; then
    echo "$HOME/.zvm/$pin/zig"
    return
  fi
  if command -v zig >/dev/null 2>&1; then
    local path_zig ver
    path_zig="$(command -v zig)"
    ver="$("$path_zig" version 2>/dev/null || true)"
    if [[ -n "$pin" && "$ver" != "$pin" ]]; then
      echo "warning: PATH zig is $ver but .zigversion pins $pin; results may not build/compare cleanly" >&2
    fi
    echo "$path_zig"
    return
  fi
  echo "error: no zig found (set ZIG=/path/to/zig)" >&2
  exit 2
}

# --- run the suite (unless --compare-only) ---
if [[ "$COMPARE_ONLY" -eq 0 ]]; then
  ZIG_BIN="$(resolve_zig)"
  RUNS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/abi-bench.XXXXXX")"
  trap 'rm -rf "$RUNS_DIR"' EXIT
  echo "== running benchmark suite x$RUNS ($("$ZIG_BIN" version)) =="
  for ((run_i = 1; run_i <= RUNS; run_i++)); do
    # Remove the old artifact so a cached/skipped run step is detectable
    # instead of silently comparing stale numbers.
    rm -f "$ARTIFACT"
    # The 'benchmarks' run step is cached by the zig build system: a repeat
    # invocation with an unchanged tree skips execution entirely and never
    # rewrites the artifact (verified: --seed does not bust it). Dropping the
    # local cache-manifest dir forces the run step to re-execute while compiled
    # objects in .zig-cache/o are still reused (~seconds, not a full rebuild).
    # Note: do not run this concurrently with another zig build in the same
    # checkout — besides racing the manifest dir, parallel builds would skew
    # benchmark timings anyway.
    rm -rf "$REPO_ROOT/.zig-cache/h"
    echo "-- suite run $run_i/$RUNS"
    (cd "$REPO_ROOT" && "$ZIG_BIN" build benchmarks)
    if [[ ! -f "$ARTIFACT" ]]; then
      echo "error: benchmark artifact not found at $ARTIFACT after run $run_i" >&2
      echo "       The 'zig build benchmarks' run step may have been cached and skipped." >&2
      echo "       Clear .zig-cache and re-run." >&2
      exit 1
    fi
    cp "$ARTIFACT" "$RUNS_DIR/run_$run_i.json"
  done

  # Aggregate best-of-N: per benchmark label, keep the whole record from the
  # run where $METRIC was smallest. Overwrites the artifact in place so the
  # baseline/compare stages (and any other artifact consumer) see the
  # aggregated numbers.
  python3 - "$ARTIFACT" "$METRIC" "$RUNS_DIR"/run_*.json <<'PY'
import json, sys
out_path, metric, run_paths = sys.argv[1], sys.argv[2], sys.argv[3:]
best = {}
order = []
meta = None
for p in run_paths:
    with open(p) as f:
        art = json.load(f)
    if art.get("schema") != "abi-bench/v2":
        sys.exit(f"error: unexpected artifact schema {art.get('schema')!r} in {p} "
                 "(expected abi-bench/v2); update tools/bench_regress.sh for the new format")
    meta = art
    for b in art["benchmarks"]:
        label = b["label"]
        if label not in best:
            order.append(label)
            best[label] = b
        elif b[metric] < best[label][metric]:
            best[label] = b
merged = {
    "schema": meta["schema"],
    "iterations": meta.get("iterations"),
    "runs_aggregated": len(run_paths),
    "aggregation": f"best {metric} of {len(run_paths)} suite runs",
    "benchmarks": [best[label] for label in order],
}
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)
    f.write("\n")
print(f"aggregated {len(run_paths)} run(s) -> {out_path} ({len(order)} benchmarks)")
PY
fi

if [[ ! -f "$ARTIFACT" ]]; then
  echo "error: benchmark artifact not found at $ARTIFACT" >&2
  echo "       Run without --compare-only first (or run 'zig build benchmarks')." >&2
  exit 1
fi

if [[ "$UPDATE_BASELINE" -eq 1 ]]; then
  python3 - "$ARTIFACT" "$BASELINE" <<'PY'
import json, platform, re, subprocess, sys
artifact_path, baseline_path = sys.argv[1], sys.argv[2]
with open(artifact_path) as f:
    art = json.load(f)
if art.get("schema") != "abi-bench/v2":
    sys.exit(f"error: unexpected artifact schema {art.get('schema')!r} (expected abi-bench/v2); "
             "update tools/bench_regress.sh for the new format")
commit = "unknown"
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    pass
# Record which metric selected the kept record during best-of-N aggregation,
# so compares can refuse a mismatched ABI_BENCH_METRIC (comparing p50 numbers
# out of min-selected records measures aggregation skew, not regressions).
selection_metric = None
m = re.match(r"best (\w+) of ", art.get("aggregation", ""))
if m:
    selection_metric = m.group(1)
baseline = {
    "schema": "abi-bench-baseline/v1",
    "source_schema": art["schema"],
    "iterations": art.get("iterations"),
    "runs_aggregated": art.get("runs_aggregated", 1),
    "aggregation": art.get("aggregation", "single run"),
    "selection_metric": selection_metric,
    "host": platform.platform(),
    "machine": platform.machine(),
    "system": platform.system(),
    "commit": commit,
    "benchmarks": art["benchmarks"],
}
with open(baseline_path, "w") as f:
    json.dump(baseline, f, indent=2)
    f.write("\n")
print(f"baseline updated: {baseline_path} ({len(art['benchmarks'])} benchmarks, commit {commit})")
PY
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "error: no baseline at $BASELINE" >&2
  echo "       Generate one first:  tools/bench_regress.sh --update-baseline" >&2
  exit 1
fi

# --- compare current artifact vs baseline ---
python3 - "$ARTIFACT" "$BASELINE" "$THRESHOLD_PCT" "$METRIC" <<'PY'
import json, os, platform, re, sys
artifact_path, baseline_path, threshold_pct, metric = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
with open(artifact_path) as f:
    art = json.load(f)
with open(baseline_path) as f:
    base = json.load(f)
if art.get("schema") != "abi-bench/v2":
    sys.exit(f"error: unexpected artifact schema {art.get('schema')!r} (expected abi-bench/v2); "
             "update tools/bench_regress.sh for the new format")

# Timing deltas across different hardware measure the machine, not the code.
# Refuse cross-host compares unless explicitly overridden.
cur_machine, cur_system = platform.machine(), platform.system()
base_machine = base.get("machine", "")
# Older baselines lack "system"; fall back to the OS prefix of "host"
# (platform.platform() output like "macOS-27.0-arm64").
base_system = base.get("system") or base.get("host", "").split("-")[0]
system_alias = {"macOS": "Darwin", "Darwin": "Darwin"}
same_system = system_alias.get(base_system, base_system) == system_alias.get(cur_system, cur_system)
if (base_machine and base_machine != cur_machine) or (base_system and not same_system):
    msg = (f"error: baseline host mismatch: baseline is {base_system}/{base_machine} "
           f"(host {base.get('host', '?')}), current is {cur_system}/{cur_machine}.\n"
           "       Cross-host timing deltas reflect hardware, not regressions.\n"
           "       Regenerate a local baseline (--update-baseline) or set "
           "ABI_BENCH_ALLOW_CROSS_HOST=1 to override.")
    if os.environ.get("ABI_BENCH_ALLOW_CROSS_HOST") == "1":
        print("warning: comparing across hosts (ABI_BENCH_ALLOW_CROSS_HOST=1); "
              "deltas include hardware differences", file=sys.stderr)
    else:
        sys.exit(msg)

# The baseline's records were selected by one metric during best-of-N
# aggregation; comparing a different field against them measures aggregation
# skew. Refuse the mismatch instead of producing a misleading verdict.
base_sel = base.get("selection_metric")
if base_sel is None:
    m = re.match(r"best (\w+) of ", base.get("aggregation", ""))
    if m:
        base_sel = m.group(1)
if base_sel and base_sel != metric:
    sys.exit(f"error: baseline records were selected by {base_sel!r} but ABI_BENCH_METRIC={metric!r}.\n"
             f"       Regenerate the baseline under this metric "
             f"(ABI_BENCH_METRIC={metric} tools/bench_regress.sh --update-baseline) "
             f"or compare with ABI_BENCH_METRIC={base_sel}.")

base_by_label = {b["label"]: b for b in base.get("benchmarks", [])}
cur_by_label = {b["label"]: b for b in art.get("benchmarks", [])}
if not cur_by_label:
    sys.exit("error: current artifact contains zero benchmarks; refusing to PASS an empty run.")

regressions = []
worst = None  # (delta_pct, label)
print(f"== comparing {metric} against baseline (threshold +{threshold_pct:g}%, "
      f"baseline commit {base.get('commit', '?')}) ==")
for label, cur in cur_by_label.items():
    ref = base_by_label.get(label)
    if ref is None:
        print(f"  NEW      {label}: {cur[metric]:.4f} ms (not in baseline; "
              "run --update-baseline to record it)")
        continue
    ref_v, cur_v = ref[metric], cur[metric]
    if ref_v <= 0:
        print(f"  SKIP     {label}: baseline {metric} is {ref_v}; cannot compute delta")
        continue
    delta_pct = (cur_v - ref_v) / ref_v * 100.0
    if worst is None or delta_pct > worst[0]:
        worst = (delta_pct, label)
    status = "OK"
    if delta_pct > threshold_pct:
        status = "REGRESS"
        regressions.append((label, ref_v, cur_v, delta_pct))
    print(f"  {status:<8} {label}: baseline {ref_v:.4f} ms -> current {cur_v:.4f} ms "
          f"({delta_pct:+.1f}%)")

missing = [label for label in base_by_label if label not in cur_by_label]
for label in missing:
    print(f"  MISSING  {label}: in baseline but absent from current run "
          "(renamed or removed benchmark?)")

print(f"-- {len(cur_by_label)} benchmarks compared; "
      f"max delta {worst[0]:+.1f}% ({worst[1]})" if worst else "-- no comparable benchmarks")
failed = False
if regressions:
    failed = True
    print(f"\nFAIL: {len(regressions)} benchmark(s) more than {threshold_pct:g}% slower than baseline:")
    for label, ref_v, cur_v, delta_pct in regressions:
        print(f"  - {label}: {ref_v:.4f} ms -> {cur_v:.4f} ms ({delta_pct:+.1f}%)")
    print("If this slowdown is expected/intentional, rerun with --update-baseline.")
if missing:
    # A vanished benchmark silently shrinks the gate; renames/removals must be
    # acknowledged by refreshing the baseline, not waved through.
    failed = True
    print(f"\nFAIL: {len(missing)} baseline benchmark(s) missing from the current run: "
          + ", ".join(missing))
    print("If the rename/removal is intentional, rerun with --update-baseline.")
if failed:
    sys.exit(1)
print("PASS: no benchmark regressions beyond threshold.")
PY
