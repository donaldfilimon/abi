#!/bin/sh
set -euo

echo "Verifying shell scripts syntax (sh -n) and DOS line endings..."
ERR=0
find tools -type f -name '*.sh' -print0 | while IFS= read -r -d '' f; do
  if ! /bin/sh -n "$f" 2>/dev/null; then
    echo "Syntax error in $f" >&2
    ERR=1
  fi
  if grep -U -q $'\r' "$f" 2>/dev/null; then
    echo "DOS line endings detected in $f" >&2
    ERR=1
  fi
done
if [ "$ERR" -eq 0 ]; then
  echo "Verification passed."
  exit 0
else
  echo "Verification failed." >&2
  exit 1
fi
