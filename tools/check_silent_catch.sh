#!/usr/bin/env bash
# Fail on silent error-discard patterns (`catch {}` / `catch { }` / `catch |_| {}`,
# including forms split across lines) in Zig sources. Wired into `zig build check`
# so swallowed errors cannot land unnoticed on persistence/inference/connector/
# data paths (see AGENTS.md).
#
# Allowlist: tools/silent_catch_allow.txt (optional) — one substring per line
# (e.g. `src/foo.zig:42` or any `file:line-pattern` fragment); blank lines and
# `#` comments ignored. A hit is skipped if any allowlist line is a substring
# of its `file:line:text` grep output.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALLOW_FILE="$ROOT/tools/silent_catch_allow.txt"

cd "$ROOT"

# Slurp-mode scan (perl -0777): \s matches newlines, so brace-split forms like
# `catch |_| {` ... `}` with a whitespace-only body are caught, not just
# single-line `catch {}`. Reports file:line:normalized-match per hit.
ALL="$({ find src tests -name '*.zig' -print0 2>/dev/null
         find tools -maxdepth 1 -name '*.zig' -print0 2>/dev/null; } |
  xargs -0 perl -0777 -ne '
  while (/catch\s*(?:\|\s*_\s*\|\s*)?\{\s*\}/g) {
    my $line = 1 + (substr($_, 0, $-[0]) =~ tr/\n//);
    my $text = substr($_, $-[0], $+[0] - $-[0]);
    $text =~ s/\s+/ /g;
    print "$ARGV:$line:$text\n";
  }' 2>/dev/null || true)"

if [[ -z "$ALL" ]]; then
  exit 0
fi

FILTERED=""
while IFS= read -r hit; do
  [[ -z "$hit" ]] && continue
  allowed=0
  if [[ -f "$ALLOW_FILE" ]]; then
    while IFS= read -r allow; do
      [[ -z "$allow" || "$allow" == \#* ]] && continue
      if [[ "$hit" == *"$allow"* ]]; then
        allowed=1
        break
      fi
    done < "$ALLOW_FILE"
  fi
  if [[ "$allowed" -eq 0 ]]; then
    FILTERED="${FILTERED:+$FILTERED$'\n'}$hit"
  fi
done <<< "$ALL"

if [[ -n "$FILTERED" ]]; then
  echo "check_silent_catch: silent error discard found (fix it, or allowlist in tools/silent_catch_allow.txt):" >&2
  printf '%s\n' "$FILTERED" >&2
  exit 1
fi

exit 0
