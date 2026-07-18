#!/usr/bin/env bash
# Validate .agents/skills/*/SKILL.md frontmatter.
# Usage: tools/check_skills.sh [skill-name]   (no arg = check all canonical skills)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/.agents/skills"
pass=0; fail=0

desc_of() { # file -> description value (single-line, or YAML folded '>' / '|')
  local f="$1"
  awk '
    /^---/{fm++; next}
    fm!=1{next}
    /^description:[ ]*/{
      v=substr($0, index($0,":")+1); sub(/^ */,"",v)
      if (v==">"||v=="|"||v==">-"||v=="|-"){fold=1; next}
      print v; exit
    }
    fold{ if($0~/^[[:space:]]/){gsub(/^[[:space:]]+/,"");print;exit} else {print "";exit} }
  ' "$f"
}

check_one() {
  local name="$1"
  local f="$DIR/$name/SKILL.md"
  if [ ! -f "$f" ]; then echo "FAIL $name: missing SKILL.md"; fail=$((fail+1)); return; fi
  local nm desc
  nm=$(awk -F': ' '/^name:/{print $2; exit}' "$f")
  desc=$(desc_of "$f")
  if [ -z "$nm" ] || [ "$nm" != "$name" ]; then
    echo "FAIL $name: name='$nm' must equal dir basename"; fail=$((fail+1)); return; fi
  if [ -z "$desc" ] || [ "${#desc}" -lt 20 ]; then
    echo "FAIL $name: description missing or <20 chars (${#desc})"; fail=$((fail+1)); return; fi
  echo "PASS $name (${#desc}c)"; pass=$((pass+1))
}

if [ "$#" -ge 1 ] && [ -n "${1:-}" ]; then
  check_one "$1"
else
  for d in "$DIR"/*/; do
    n=$(basename "$d"); [ "$n" = "sync-clis" ] && continue
    [ -d "$d" ] && check_one "$n"
  done
fi
echo "=== summary: pass=$pass fail=$fail ==="
[ "$fail" -eq 0 ]
