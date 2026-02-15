# Repository Guidelines

**Contents:** [Project Structure](#project-structure--module-organization) · [Build & Test](#build-test-and-development-commands) · [Coding Style](#coding-style--naming-conventions) · [Testing](#testing-guidelines) · [Commits & PRs](#commit--pull-request-guidelines) · [Security](#security--configuration-notes)

## Project Structure & Module Organization

ABI is a Zig 0.16 framework (v0.4.0) with `src/abi.zig` as the public API entrypoint. Core implementation is split across:

- `src/api/`, `src/core/`, `src/services/`, and `src/features/` for primary code.
- **21 feature modules** (each with `mod.zig` + `stub.zig`): `ai`, `ai_core`, `ai_inference`, `ai_training`, `ai_reasoning`, `analytics`, `auth`, `benchmarks`, `cache`, `cloud`, `database`, `gateway`, `gpu`, `messaging`, `mobile`, `network`, `observability`, `pages`, `search`, `storage`, `web`.
- **Services**: `connectors/` (9 LLM providers + discord + scheduler), `mcp/` (JSON-RPC 2.0 server), `acp/` (agent protocol), `runtime/`, `shared/`, `ha/`, `platform/`, `tasks/`.
- **28 CLI commands** + 8 aliases (`info`, `sysinfo`, `ls`, `run`, `dashboard`, `chat`, `reasoning`, `serve`).
- Shared runtime and integration tests: `src/services/tests/`.
- Feature inline tests: `src/feature_test_root.zig`.
- Examples/docs: `examples/`, `benchmarks/`, `docs/api/`.

## Build, Test, and Development Commands

- `zig build`: build with default feature flags.
- `zig build run -- --help`: show CLI help (28 commands).
- `zig build run -- system-info`: system and feature status.
- `zig build run -- mcp serve`: start MCP server (stdio JSON-RPC).
- `zig build run -- acp card`: print ACP agent card.
- `zig build test --summary all`: run the full test suite (baseline: 1248 pass, 5 skip).
- `zig build feature-tests --summary all`: run feature module inline tests (baseline: 977 pass).
- `zig test src/path/to/file.zig --test-filter "pattern"`: run focused tests.
- `zig build validate-flags`: validate 34 feature-flag combinations.
- `zig build cli-tests`: run CLI smoke tests (top-level and nested commands).
- `zig build full-check`: local quality gate (format + tests + flags + CLI smoke).
- `zig fmt .`: format sources before PR review.
- `zig build lint`: formatting/lint check used in CI.

## Coding Style & Naming Conventions

- Use Zig `0.16.0-dev.2596+469bf6af0` or newer.
- Indentation: 4 spaces, no tabs.
- Naming: `PascalCase` types, `camelCase` functions/variables.
- Prefer explicit imports (`@import(...)`) and avoid `usingnamespace`.
- Prefer `defer`/`errdefer` for deterministic cleanup.
- Prefer `std.ArrayListUnmanaged(T).empty` patterns where appropriate.
- Use `{t}` format specifier for enums and errors.
- `std.log.*` in library code; `std.debug.print` only in CLI tools.

## Testing Guidelines

- **Test baselines**: 1248/1253 main (5 skip), 977/977 feature — must be maintained.
- Unit tests should live next to implementation in `*_test.zig`.
- Integration/stress/parity tests go in `src/services/tests/`.
- Feature inline tests discovered via `src/feature_test_root.zig`.
- Hardware-dependent tests must skip with `error.SkipZigTest`.
- Test names should be descriptive (e.g., `test "Feature ..."`).
- Use `test { _ = @import(...); }` for test discovery (NOT `comptime {}`).

## Commit & Pull Request Guidelines

- Use conventional commit style (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- PRs should include:
  - clear summary of behavior changes,
  - linked issue/PR context when applicable,
  - commands run and outcomes (especially tests),
  - docs updates for API or feature-flag changes.

## Security & Configuration Notes

- Do not hardcode secrets; pass credentials via environment variables.
- When editing feature-gated modules, validate both paths:
  - `zig build -Denable-<feature>=true`
  - `zig build -Denable-<feature>=false`
- When editing `mod.zig`, always update `stub.zig` to match.

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) — Workflow and PR checklist
- [CLAUDE.md](CLAUDE.md) — Build commands, gotchas, and architecture
- [SECURITY.md](SECURITY.md) — Security policy and reporting
