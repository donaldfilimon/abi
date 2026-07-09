#!/usr/bin/env bash
# Structural gate for modern-refactor skill packaging.
# Asserts advertised references/examples exist and SKILL.md resource paths resolve.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

required=(
  skills/codebase-analysis/references/analysis-checklist.md
  skills/refactor-implementation/references/implementation-playbook.md
  skills/refactor-implementation/examples/parallel-extract-outline.md
  skills/modern-patterns/references/patterns-catalog.md
  skills/modern-patterns/examples/before-after-zig.md
  skills/refactor-strategy/references/strategy-guide.md
  skills/refactor-strategy/examples/sample-plan-outline.md
  skills/refactor-validation/references/validation-checklist.md
  .claude-plugin/plugin.json
  .claude/modern-refactor.local.md.example
  README.md
)

fail=0
for f in "${required[@]}"; do
  if [ ! -f "$f" ]; then
    echo "missing required file: $f" >&2
    fail=1
  fi
done

# Every SKILL.md line like `- \`references/...\`` or `- \`examples/...\`` must resolve.
while IFS= read -r skill_md; do
  skill_dir="$(dirname "$skill_md")"
  while IFS= read -r line; do
    case "$line" in
      "- \`references/"*|"- \`examples/"*)
        path="${line#- \`}"
        path="${path%%\`*}"
        path="${path%% *}"
        if [ -z "$path" ]; then
          continue
        fi
        target="$skill_dir/$path"
        if [ ! -e "$target" ]; then
          echo "SKILL resource missing: $skill_md -> $path" >&2
          fail=1
        fi
        ;;
    esac
  done < "$skill_md"
done < <(find skills -name SKILL.md -print | sort)

# README must not claim in-tree PreToolUse hooks as packaged features.
if grep -q '^\- \*\*PreToolUse hooks\*\*' README.md 2>/dev/null; then
  echo "README still advertises packaged PreToolUse hooks" >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "verify-plugin-layout: FAIL" >&2
  exit 1
fi
echo "verify-plugin-layout: OK ($ROOT)"
