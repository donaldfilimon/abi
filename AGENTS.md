# Repository Guidelines

## Project Structure & Module Organization
- `src/abi.zig`: public API and comptime feature gating.
- `src/features/<name>/`: paired `mod.zig` + `stub.zig`; keep signatures aligned.
- `src/core/`: lifecycle, config, registry; `src/services/`: shared runtime infrastructure.
- `tools/cli/`: CLI framework (`main.zig`, `commands/`, `spec.zig`, `utils/`) and TUI framework (`tui/`).
- `build/`: modular build graph; `docs/`, `examples/`, `tools/scripts/`: docs, runnable samples, quality gates.
- `src/services/tests/`: integration/system tests; unit tests should stay near the module.

## Build, Test, and Development Commands
- `zig build`: build/install `abi`.
- `zig build run -- --help`: run CLI locally; `zig build run -- tui` opens TUI.
- `zig build test --summary all`: main tests.
- `zig build feature-tests --summary all`: feature inline tests.
- `zig build cli-tests`: CLI smoke tests.
- `zig build validate-flags`: feature-flag matrix validation.
- `zig fmt .`: format sources.
- `zig build full-check`: required PR gate.
- `zig build verify-all`: extended gate (`full-check` + consistency + examples + wasm).
- `zig build toolchain-doctor`: diagnose local Zig mismatch.

## Coding Style & Naming Conventions
- Zig version is pinned in `.zigversion`.
- 4 spaces, no tabs, target lines under ~100 chars.
- Types/enums: `PascalCase`; functions/variables: `camelCase`; files: `snake_case.zig`.
- Use explicit imports; avoid `usingnamespace`.
- Prefer specific error sets over `anyerror`, and use `{t}` for enum/error formatting.
- For file/network I/O, initialize `std.Io.Threaded` and use `std.Io.Dir.cwd()`.

## CLI & TUI Framework Rules
- New command flow: implement in `tools/cli/commands/`, export in `tools/cli/commands/mod.zig`, then register help/completion metadata in `tools/cli/spec.zig`.
- I/O-heavy commands should follow the `std.Io` path used by `tools/cli/mod.zig`.
- Keep TUI work modular in `tools/cli/tui/` (panels, widgets, themes, async loop).

## Testing, Commits, and PRs
- Add tests for every behavior change.
- If `mod.zig` changes, update sibling `stub.zig` and test `-Denable-<feature>=true/false`.
- If test totals change, update `tools/scripts/baseline.zig` and run `zig build validate-baseline`.
- Use Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Keep PRs focused and include motivation, linked issue(s), commands run (usually `zig build full-check`), and docs updates for API/CLI changes.

## Agent Workflow Notes
- Follow `CLAUDE.md` for deep architecture and gotchas.
- In direct editor/chat sessions, work outside the Ralph loop: edit, run local checks, and validate with `zig build full-check`.
- Use `abi ralph ...` only when explicitly requested for iterative runs.

## Security & Configuration Tips
- Do not commit secrets or local credentials.
- Report vulnerabilities via `SECURITY.md` (avoid public exploit details in issues/PRs).

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
