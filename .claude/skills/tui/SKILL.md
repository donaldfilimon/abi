---
name: tui
description: Plan abi TUI/dashboard work — the interactive diagnostics dashboard and agent REPL. Use when asked about abi tui / dashboard, pane splits, slash commands, or session save/load. Routes to run-tui, dashboard-smoke, and abi-superpower-tui. A headless fallback exists; tmux is only for the interactive refresh loop.
---

# tui

Entry point for abi's TUI surface (`abi tui` / `abi dashboard` / `abi --tui`).
Routes:

| You want to… | Use |
| --- | --- |
| Drive the interactive dashboard in a real pty (screenshot) | `run-tui` |
| Non-interactive one-shot `abi dashboard` smoke (CI/headless) | `dashboard-smoke` |
| Deep-dive the TUI superpower (panes, slash commands) | `abi-superpower-tui` |

## Slash commands backed by skills (`.opencode.json` `slash_commands`)
`/open`→file-context-loader, `/diff`→git-diff-integration,
`/commit`→git-commit-integration, `/context`→context-state-reporter,
`/features`→feature-flag-display, `/learn`→sea-learning-controller,
`/save`→session-persister, `/load`→session-restorer,
`/status`→agent-status-reporter, `/reset`→context-resetter. Plugin-provided
commands come from `abi-plugin.json` `commands`.

## Gotchas
- `dashboard-smoke` reads stdin from `/dev/null` to force the non-interactive
  fallback; the only surface that needs a real terminal is the interactive
  refresh loop (`run-tui` uses tmux).
- `@file` mentions are sandboxed to cwd (8 KB budget; rejects `..` / absolute /
  symlink escape) via `file_context.zig`.
