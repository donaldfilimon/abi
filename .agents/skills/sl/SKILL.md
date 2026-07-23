---
name: sl
description: Manage and maintain AI coding skills via the skill-loop MCP server. Scan, inspect, review, fix, and monitor skill health.
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
      "args": ["-y", "-p", "@stylusnexus/skill-loop-cli", "skill-loop-mcp"]
    }
  }
}
```

Note: npx-based MCPs may fail handshake in some envs (see startup diagnostics). Alternatives or local installs can be configured in `.mcp.json`. After editing MCP config, restart the Grok session/TUI.

**CLI vs MCP:** There is no in-repo `skill-loop` binary. The CLI ships as
`@stylusnexus/skill-loop-cli` (repo pins `@0.3.3` in `.mcp.json`). Bare
`skill-loop` on `PATH` is optional — use
`npx -y -p @stylusnexus/skill-loop-cli@0.3.3 skill-loop <cmd>` or
`npm i -g @stylusnexus/skill-loop-cli@0.3.3`. CLI subcommands differ from the
MCP `/sl` actions above: rescan is `init` (not `scan`); analysis is `inspect`;
logging is `skill-loop log <skill-name> <outcome>`.

## Maintenance
During 2026-07-06 "do all" / hygiene pass, used skill-loop MCP directly for review + amend + apply_fixes on registered skills (fixed broken file refs in sl and abbey-assistant). Full-check now passes (45/45). Added cross-refs in other skills. Re-scan after adding new SKILL.md. See ~/CLEANUP-2026-07-06.md.
