---
name: sl
description: >-
  Audit and repair the health of installed SKILL.md files through the skill-loop
  MCP server. Use when the user runs /sl, or asks to scan and register skills,
  check skill health or failure rate, review skills for staleness or broken file
  references, propose or apply fixes for a broken skill, or see recent skill
  runs, past amendments, or detection stats. Do NOT use to author a new skill
  (use create-skill), to answer Grok configuration questions (use help), or to
  propagate skills to other CLIs (use sync-clis).
---

# sl (skill-loop)

Route the user's request to the `skill_loop` MCP tool. This skill is a convenience wrapper -- all actions are handled by the MCP server.

## Usage

`/sl <action>` where action is one of:

| Action | What it does |
|--------|-------------|
| `scan` | Scan for SKILL.md files and register them |
| `status` | Health dashboard: skill count, runs, failure rate |
| `review` | Analyze all skills for failure patterns and staleness |
| `fix` | Propose amendments for broken skills (creates a git branch) |
| `fix --dry-run` | Preview fixes without modifying anything |
| `list` | Show all registered skills with metadata |
| `runs` | Show recent skill run activity |
| `history` | List past amendments and their status |
| `detection` | Show detection stats and active sessions |
| `gc` | Prune old run data |

## How to handle

1. Parse the action from the user's input (everything after `/sl`)
2. Call the `skill_loop` MCP tool with `action` set to the parsed action
3. Present the results to the user

If no action is provided, default to `status`.

If the MCP server is not connected, tell the user to add this to their `.mcp.json`:

```json
{
  "mcpServers": {
    "skill-loop": {
      "command": "npx",
      "args": ["-y", "-p", "@stylusnexus/skill-loop-cli@0.3.3", "skill-loop-mcp"]
    }
  }
}
```

Note: npx-based MCPs may fail handshake in some envs (see startup diagnostics). Alternatives or local installs can be configured in `.mcp.json`. After editing MCP config, restart the Grok session/TUI. Use `skill-loop` to scan/review/fix skill health across all folders.

## Maintenance

Re-run `/sl scan` after adding or moving any SKILL.md — the registry does not
watch the filesystem.
