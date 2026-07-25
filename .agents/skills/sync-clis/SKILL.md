---
name: sync-clis
description: Sync canonical skills/plugins/commands/experiences from central (.grok + abi-mega) to all CLIs (grok,claude,codex,opencode,abi,cursor+). Idempotent. Launch with /sync-clis or the launch.sh .
---
# /sync-clis

This skill is backed by the executable launcher at
`.agents/skills/sync-clis/launch.sh` (run via Grok skill system or directly).

It syncs canonical `.agents/skills/` into in-repo `.claude/skills/` and `.grok/`
(see launcher header; distinct from `~/.grok/scripts/sync-clis.py`).
