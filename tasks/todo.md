# ABI Zig Improvement Implementation Plan (Track A First)

## Objective
Implement the approved reliability-first plan by stabilizing CLI/build health now (Track A), then advancing broader improvement work (Track B) without disrupting existing in-progress repository work.

## Scope
- Track A (now): unblock CLI compile failures and prove MCP/LSP/CLI command paths.
- Track A (now): run stabilization gates (`toolchain-doctor`, `typecheck`, command help paths, `cli-tests`, broader checks).
- Track B (next): start reliability hardening with targeted `catch {}` cleanup and command help consistency follow-up.

## Verification Criteria
- `zig build run -- mcp --help` passes.
- `zig build run -- mcp serve --zls --help` passes.
- `zig build run -- mcp tools --help` passes.
- `zig build run -- lsp --help` passes.
- `zig build cli-tests` passes.
- `zig build typecheck` remains passing.

## Checklist
- [ ] Fix duplicate local variable in `tools/cli/commands/os_agent.zig`.
- [ ] Fix top-level help wiring in `tools/cli/framework/help.zig` (missing symbol/allocator issues).
- [ ] Fix `std` import visibility issue in `tools/cli/tui/keybindings.zig` tests.
- [ ] Run `zig build toolchain-doctor`.
- [ ] Run `zig build typecheck`.
- [ ] Run CLI command-surface checks for MCP/LSP.
- [ ] Run `zig build cli-tests`.
- [ ] Run broader gate (`zig build full-check`) if fast enough after unblock.
- [ ] Record outcomes and residual risk.

## Review
- In progress.
