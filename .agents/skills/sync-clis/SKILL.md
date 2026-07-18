---
name: sync-clis
description: Sync canonical skills from abi `.agents/skills/` to in-repo CLI skill dirs (`.claude/skills`, `.grok`). Idempotent. Launch with /sync-clis or launch.sh. Distinct from home `~/.grok/scripts/sync-clis.py`.
---
# /sync-clis

This skill is backed by `launch.sh` in this directory (run via Grok skill system or directly).

## What abi `launch.sh` copies today

- **Canonical source:** repo `.agents/skills/<name>/`
- **Targets (if present):** repo `.claude/skills/`, repo `.grok/`
- **Per skill (creates the target dir if missing):**
  - `SKILL.md` (rewrites `Base directory for this skill:` to the target path)
  - `references/` (when present at source)
  - `examples/` (when present at source)
- **Not copied:** launcher `.sh` scripts, other skill payloads, skills on the skip list.

Skip list (not synced by this launcher): `abi-doc-claims-sync`, `abi-goal-orchestrator`, `check-work`, `code-review`, `create-skill`, `docx`, `help`, `imagine`, `pptx`, `sl`, `sync-clis`, `xlsx`.

## Limitation / related sync

Home multi-CLI sync (`~/.claude/skills/sync-clis` → `~/.grok/scripts/run-sync-clis.sh` → `sync-clis.py`) is a **different** path: central `.grok/skills` (+ abi-mega seed) → configured CLI targets. That driver still copies **`SKILL.md` only** (plus persona/command wrappers). Do not claim home sync keeps `references/` or `examples/` in parity unless `sync-clis.py` is extended separately.
