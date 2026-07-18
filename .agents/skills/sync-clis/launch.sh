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

# Print would-msg under --dry-run; otherwise run the command and print done-msg.
run_or_echo() {
    local would_msg="$1" done_msg="$2"
    shift 2
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  $would_msg"
    else
        "$@"
        echo "  $done_msg"
    fi
}

sync_skills() {
    local src="$1" dst="$2"
    local label="$3"

    # Skip universal skills listed below; sync the rest (create target dirs if missing).
    for skill_dir in "$src"/*/; do
        local skill_name
        skill_name=$(basename "$skill_dir")

        case "$skill_name" in
            abi-doc-claims-sync|abi-goal-orchestrator|check-work|code-review|create-skill|docx|help|imagine|pptx|sl|sync-clis|xlsx)
                continue
                ;;
        esac

        if [ ! -d "$dst/$skill_name" ]; then
            run_or_echo "would create: $label/$skill_name/" "created: $label/$skill_name/" \
                mkdir -p "$dst/$skill_name"
        fi

        # Sync SKILL.md (text content only, not the .sh scripts).
        # Rewrite "Base directory for this skill:" to the target path (not canonical).
        if [ -f "$src/$skill_name/SKILL.md" ]; then
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "  would sync: $label/$skill_name/SKILL.md"
            else
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
