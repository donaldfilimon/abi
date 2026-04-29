#!/usr/bin/env bash
set -euo pipefail

# Canonical root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Doc Cross-Link Validation ==="
declare -a failures=()

if ! grep -q -E 'GLOSSARY\.md|ONBOARDING' ONBOARDING.md; then
  echo "Warning: ONBOARDING.md does not reference GLOSSARY.md"
  failures+=("onboarding_missing_glossary_link")
fi
if ! grep -q -E 'CODEBASE_REVIEW\.md|ONBOARDING' ONBOARDING.md; then
  echo "Warning: ONBOARDING.md does not reference CODEBASE_REVIEW.md"
  failures+=("onboarding_missing_codebase_link")
fi

if ! grep -q -E 'ONBOARDING\.md|GLOSSARY\.md' CODEBASE_REVIEW.md; then
  echo "Warning: CODEBASE_REVIEW.md lacks ONBOARDING.md or GLOSSARY.md references"
  failures+=("codebase_missing_links")
fi

if ! grep -q -E 'ONBOARDING\.md|ONBOARDING' GLOSSARY.md; then
  echo "Warning: GLOSSARY.md does not reference ONBOARDING.md"
  failures+=("glossary_missing_onboarding_link")
fi

echo "Checking Mermaid blocks balance..."
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  if grep -q '^```mermaid' "$f"; then
    in=0
    while IFS= read -r line; do
      if [[ "$line" == '```mermaid' ]]; then
        if (( in == 1 )); then
          echo "Unclosed mermaid block detected in $f"
          failures+=("mermaid_block_unclosed:$f")
          break
        fi
        in=1
      elif [[ "$line" == '```' ]]; then
        if (( in == 1 )); then
          in=0
        fi
      fi
    done < "$f"
    if (( in == 1 )); then
      echo "Unclosed mermaid block detected in $f"
      failures+=("mermaid_block_unclosed:$f")
    fi
  fi
done < <(find . -type f -name '*.md' -print0)

if (( ${#failures[@]} > 0 )); then
  echo "Documentation validation FAILED with ${#failures[@]} issue(s)."
  exit 1
fi

echo "Documentation validation PASSED."
