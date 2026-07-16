---
name: opencode
description: Set up and operate abi through opencode — how opencode discovers abi skills, slash-commands, and MCP servers. Use when the user asks about opencode config, why an abi skill is not loading in opencode, or how to add a skill to the opencode allowlist.
---

# opencode

opencode loads abi via three mechanisms wired in `.opencode.json` + `opencode.json`:

- **Skills**: `.opencode/skills` is a symlink to `../.agents/skills`, so every
  canonical skill dir is visible to opencode automatically. The `.opencode.json`
  `skills` array is the *allowlist* of which skills opencode advertises. To expose
  a new skill in opencode, add its dir basename to that array — no file copy.
- **Slash commands**: `.opencode.json` `slash_commands` maps `/open /diff /commit
  /context /features /learn /save /load /status /reset` to backing skills
  (`file-context-loader`, `git-diff-integration`, `git-commit-integration`,
  `context-state-reporter`, `feature-flag-display`, `sea-learning-controller`,
  `session-persister`, `session-restorer`, `agent-status-reporter`,
  `context-resetter`). Add a command by pointing `skill:` at a canonical name.
- **MCP servers**: `opencode.json` `mcp` wires `abi-mcp` (via
  `./mcp/launcher.sh stdio`) and `skill-loop` (the telemetry/registry engine).
  `.mcp.json` mirrors the same for other clients.

## To add a skill to opencode
1. Author/complete the skill under `.agents/skills/<name>/SKILL.md`.
2. Add `"<name>"` to the `.opencode.json` `skills` array.
3. Run `tools/check_skills.sh <name>` to confirm frontmatter is valid.
4. Skills auto-reload from disk — no restart needed.

## Gotchas
- `.opencode/skills` is a symlink; editing a skill there edits the canonical copy.
- `skill-loop` writes `.skill-telemetry/registry.json` + `amendments.jsonl`
  automatically (content-drift proposals). Do not hand-edit the registry.
- The `sync-clis/launch.sh` syncer does NOT create new target dirs in
  `.claude/skills/`/`.grok/`; it only updates `SKILL.md` for dirs that already
  exist there. opencode needs no sync (symlink), but Claude Code/grok do.
