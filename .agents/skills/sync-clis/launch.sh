#!/usr/bin/env bash
# sync-clis: sync canonical skills from .agents/skills/ to in-repo CLI skill
# dirs. Idempotent. Copies SKILL.md plus references/ and examples/ when present.
# Does not copy .sh launchers. Distinct from ~/.grok/scripts/sync-clis.py.
#
# Usage:
#   .agents/skills/sync-clis/launch.sh              # sync all targets
#   .agents/skills/sync-clis/launch.sh --dry-run    # preview only
#
# Targets (if present): .claude/skills/, .grok/
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

DRY_RUN=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        --help) echo "Usage: $(basename "$0") [--dry-run]"; exit 0 ;;
        *) echo "unknown flag: $a" >&2; exit 2 ;;
    esac
done

CANONICAL="$REPO_ROOT/.agents/skills"
TARGETS=()
[ -d "$REPO_ROOT/.claude/skills" ] && TARGETS+=("$REPO_ROOT/.claude/skills")
[ -d "$REPO_ROOT/.grok" ]         && TARGETS+=("$REPO_ROOT/.grok")

say() { printf '\n=== %s ===\n' "$*"; }

sync_skills() {
    local src="$1" dst="$2"
    local name label="$3"

    # Sync only ABI-specific skills (those with .sh scripts or shared by all)
    for skill_dir in "$src"/*/; do
        local skill_name
        skill_name=$(basename "$skill_dir")

        # Skip universal/cross-platform skills that are only in .agents/
        case "$skill_name" in
            abi-doc-claims-sync|abi-goal-orchestrator|check-work|code-review|create-skill|docx|help|imagine|pptx|sl|sync-clis|xlsx)
                continue
                ;;
        esac

        if [ ! -d "$dst/$skill_name" ]; then
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "  would create: $label/$skill_name/"
            else
                mkdir -p "$dst/$skill_name"
                echo "  created: $label/$skill_name/"
            fi
        fi

        # Sync SKILL.md (text content only, not the .sh scripts)
        if [ -f "$src/$skill_name/SKILL.md" ]; then
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "  would sync: $label/$skill_name/SKILL.md"
            else
                # Rewrite any "Base directory for this skill: <path>" footer so it
                # points at this target's actual location, not the canonical source.
                sed "s|^Base directory for this skill: .*\$|Base directory for this skill: $dst/$skill_name|" \
                    "$src/$skill_name/SKILL.md" > "$dst/$skill_name/SKILL.md"
                echo "  synced: $label/$skill_name/SKILL.md"
            fi
        fi
        # Companion docs skills may load; non-destructive (overwrite/add, no delete)
        for sub in references examples; do
            if [ -d "$src/$skill_name/$sub" ]; then
                if [ "$DRY_RUN" -eq 1 ]; then
                    echo "  would sync: $label/$skill_name/$sub/"
                else
                    mkdir -p "$dst/$skill_name/$sub"
                    cp -R "$src/$skill_name/$sub/." "$dst/$skill_name/$sub/"
                    echo "  synced: $label/$skill_name/$sub/"
                fi
            fi
        done
    done
}

say "sync-clis: canonical=$CANONICAL"
echo "targets: ${TARGETS[*]:-none}"
echo ""

for t in "${TARGETS[@]}"; do
    label=$(basename "$(dirname "$t")")
    sync_skills "$CANONICAL" "$t" "$label"
done

say "done"
echo "Synced canonical skills to ${#TARGETS[@]} target(s)."
