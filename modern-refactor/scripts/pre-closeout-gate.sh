#!/usr/bin/env bash
# Hard prerequisite: no concurrent writers outside this process tree.
# Exits 0 if safe; exits 1 and writes blocking-writers.txt if not.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${SCRATCH:-}"
if [ -z "$SCRATCH" ]; then
  echo "pre-closeout-gate: set SCRATCH to implementer scratch dir" >&2
  exit 2
fi
mkdir -p "$SCRATCH"

# Build list of PIDs in our ancestry (self + parents) — those are NOT blockers.
ancestors=""
pid=$$
while [ -n "${pid:-}" ] && [ "$pid" -gt 1 ]; do
  ancestors="$ancestors $pid "
  pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  [ -z "${pid:-}" ] && break
done

is_ancestor() {
  case "$ancestors" in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

: > "$SCRATCH/blocking-writers.txt"
blocking=0

while IFS= read -r line; do
  [ -z "$line" ] && continue
  bpid="${line%% *}"
  if is_ancestor "$bpid"; then
    continue
  fi
  case "$line" in
    *server-github*|*skill-loop-mcp*|*modelcontextprotocol*) continue ;;
  esac
  echo "$line" >> "$SCRATCH/blocking-writers.txt"
  blocking=1
done < <(pgrep -fl 'grok --yolo|grok.*abi' 2>/dev/null || true)

if [ "$blocking" -eq 1 ]; then
  {
    echo "blocking writers at $(date -Iseconds)"
    echo "self=$$ ancestors=$ancestors"
    cat "$SCRATCH/blocking-writers.txt"
  } >&2
  echo "pre-closeout-gate: FAIL (concurrent writers)" >&2
  exit 1
fi

echo "pre-closeout-gate: OK (no foreign writers)"
rm -f "$SCRATCH/blocking-writers.txt"
exit 0
