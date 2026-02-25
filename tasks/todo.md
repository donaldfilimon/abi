# Cursor-like CLI TUI Editor Task Plan

## Objective
Add an inline CLI TUI text editor command (Cursor-like terminal workflow) to ABI, reachable via `abi editor` and a dedicated `zig build` step, using existing Zig CLI architecture.

## Scope
- Add a new CLI command module under `tools/cli/commands/`.
- Register the command in `tools/cli/commands/mod.zig`.
- Add a build step in `build.zig` for launching the editor flow.
- Keep implementation self-contained and compatible with Zig 0.16 patterns in this repo.

## Verification Criteria
- `zig build run -- editor --help` shows the new command and help output.
- `zig build editor -- --help` invokes the editor entry path via build step.
- No unresolved command wiring errors during command execution.

## Checklist
- [x] Create `editor` command module with inline TUI loop and basic file editing behavior.
- [x] Register `editor` command in the command module registry.
- [x] Add `editor` build step in `build.zig` that forwards args to `abi editor`.
- [x] Run command-surface verification commands and capture outcomes.
- [x] Mark checklist complete with evidence.

## Evidence
- `zig build run -- editor --help` succeeded and printed editor usage/help text.
- `zig build editor -- --help` succeeded and printed the same help text via the dedicated build step.
- PTY visual verification: `zig build run -- editor` rendered the TUI frame (alt-screen header, line gutter/status bar) and exited cleanly on `Ctrl-Q`.
