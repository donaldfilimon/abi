---
name: opencode
description: Set up and operate abi through opencode — how opencode discovers abi skills, slash-commands, and MCP servers. Use when the user asks about opencode config, why an abi skill is not loading in opencode, or how to add a skill to the opencode allowlist.
---

# opencode

opencode loads abi via mechanisms wired in repo-root `opencode.json` and the
`.opencode/skills` symlink:

- **Skills**: `.opencode/skills` is a symlink to `../.agents/skills`, so every
  canonical skill dir is visible to opencode automatically. No separate allowlist
  file is required today — editing a skill under `.agents/skills/<name>/` is enough.
- **Agent TUI slash commands**: live in the Rust REPL, not an OpenCode JSON
  map. The current surface is `/help`, `/model`, `/profile`, `/sea`, `/status`,
  `/context`, `/history`, `/reset`, `/features`, `/clear`, and `/quit` plus
  documented aliases. Workflow skills for Git or session planning do not add
  `/diff`, `/commit`, `/save`, or `/load` to the binary.
- **MCP servers**: `opencode.json` `mcp` wires `abi-mcp` (via
  `./mcp/launcher.sh stdio`) and `skill-loop` (the telemetry/registry engine).
  `.mcp.json` mirrors the same for other clients.
- **Instructions**: `opencode.json` `instructions` points at `AGENTS.md`,
  `tasks/lessons.md`, and `tasks/todo.md`.

## To add a skill for opencode
1. Author/complete the skill under `.agents/skills/<name>/SKILL.md`.
2. Run `tools/check_skills.sh <name>` to confirm frontmatter is valid.
3. Skills auto-reload from the symlink — no restart needed for content edits.

## Gotchas
- `.opencode/skills` is a symlink; editing a skill there edits the canonical copy.
- `skill-loop` writes `.skill-telemetry/registry.json` plus amendment proposals
  under `.skill-telemetry/` automatically (content-drift proposals). Do not
  hand-edit the registry. (`.skill-telemetry/` is gitignored.)
- The `.agents/skills/sync-clis/launch.sh` syncer updates in-repo
  `.claude/skills/` and `.grok/` from the canonical tree (creates missing skill
  dirs; skip-list skills are not copied). opencode needs no sync (symlink), but
  Claude Code / grok do.
