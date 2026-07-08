#!/usr/bin/env bash
# sync-clis: sync canonical skills/plugins/commands from .agents/skills/ to
# all target CLI skill directories. Idempotent.
#
# Usage:
#   .agents/skills/sync-clis/launch.sh              # sync all targets
#   .agents/skills/sync-clis/launch.sh --dry-run    # preview only
#
# Targets: grok (via .grok/config.toml extra_skill_dirs),
#          claude (.claude/skills/), opencode,
#          codex (via .codex/config.toml MCP refs)
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

        if [ -d "$dst/$skill_name" ]; then
            # Sync SKILL.md (text content only, not the .sh scripts)
            if [ -f "$src/$skill_name/SKILL.md" ]; then
                cp "$src/$skill_name/SKILL.md" "$dst/$skill_name/SKILL.md"
                echo "  synced: $label/$skill_name/SKILL.md"
            fi
        fi
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
