# Repository Guidelines

**Contents:** [Project Structure](#project-structure--module-organization) · [Build & Test](#build-test-and-development-commands) · [Coding Style](#coding-style--naming-conventions) · [Testing](#testing-guidelines) · [Commits & PRs](#commit--pull-request-guidelines) · [Security](#security--configuration-notes) · [Plans, Skills, and Agent Roles](#plans-skills-and-agent-roles) · [See Also](#see-also)

**Cursor / Claude:** Work *outside* the Ralph loop—edit, build, test directly. [CLAUDE.md](CLAUDE.md#working-outside-the-ralph-loop-cursor--claude) · [Skills, Plans, and Agents](CLAUDE.md#skills-plans-and-agents-full-index).

## Project Structure & Module Organization

ABI is a Zig 0.16 framework (v0.4.0) with `src/abi.zig` as the public API entrypoint. Core implementation is split across:

- `src/api/`, `src/core/`, `src/services/`, and `src/features/` for primary code.
- **21 feature modules** (each with `mod.zig` + `stub.zig`): `ai`, `ai_core`, `ai_inference`, `ai_training`, `ai_reasoning`, `analytics`, `auth`, `benchmarks`, `cache`, `cloud`, `database`, `gateway`, `gpu`, `messaging`, `mobile`, `network`, `observability`, `pages`, `search`, `storage`, `web`.
- **Services**: `connectors/` (9 LLM providers + discord + scheduler), `mcp/` (JSON-RPC 2.0 server), `acp/` (agent protocol), `runtime/`, `shared/`, `ha/`, `platform/`, `tasks/`.
- **30 CLI commands** + 8 aliases (`info`, `sysinfo`, `ls`, `run`, `dashboard`, `chat`, `reasoning`, `serve`).
- Shared runtime and integration tests: `src/services/tests/`.
- Feature inline tests: `src/feature_test_root.zig`.
- Examples/docs: `examples/`, `benchmarks/`, `docs/api/`.

## Build, Test, and Development Commands

- `zig build`: build with default feature flags.
- `zig build run -- --help`: show CLI help (30 commands).
- `zig build run -- system-info`: system and feature status (includes Feature Matrix).
- `zig build run -- --list-features`: list features (COMPILED/DISABLED) and exit.
- `zig build run -- status`: framework health and feature count.
- `zig build run -- mcp serve`: start MCP server (stdio JSON-RPC).
- `zig build run -- acp card`: print ACP agent card.
- `zig build test --summary all`: run the full test suite (baseline: 1270 pass, 5 skip).
- `zig build feature-tests --summary all`: run feature module inline tests (baseline: 1535 pass).
- `zig test src/path/to/file.zig --test-filter "pattern"`: run focused tests.
- `zig build validate-flags`: validate 34 feature-flag combinations.
- `zig build cli-tests`: run CLI smoke tests (top-level and nested commands).
- `zig build full-check`: local quality gate (format + tests + flags + CLI smoke).
- `zig build toolchain-doctor`: diagnose local Zig PATH/version drift against `.zigversion`.
- `zig fmt .`: format sources before PR review.
- `zig build lint`: formatting/lint check used in CI.
- `zig build fix`: format sources in place.

## Coding Style & Naming Conventions

- Use Zig `0.16.0-dev.2611+f996d2866` or newer.
- Indentation: 4 spaces, no tabs.
- Naming: `PascalCase` types, `camelCase` functions/variables.
- Prefer explicit imports (`@import(...)`) and avoid `usingnamespace`.
- Prefer `defer`/`errdefer` for deterministic cleanup.
- Prefer `std.ArrayListUnmanaged(T).empty` patterns where appropriate.
- Use `{t}` format specifier for enums and errors.
- `std.log.*` in library code; `std.debug.print` only in CLI tools.

## Testing Guidelines

- **Test baselines**: 1270/1275 main (5 skip), 1535/1535 feature — must be maintained.
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
- Full feature list, catalog location, and "when editing" steps: [CLAUDE.md — Feature Flags and Adding a New Feature Module](CLAUDE.md#feature-flags).

## Plans, Skills, and Agent Roles

### Multi-agent roles (from plans/plan.md)

When executing phased work (e.g. from `plans/plan.md` or a child plan), roles are:

| Role | Ownership | Responsibility | Outputs |
|------|-----------|----------------|---------|
| **A0 Coordinator** | Cross-cutting | Phase sequencing, conflict resolution, go/no-go | Status, readiness call |
| **A1 Feature Parity** | `src/features/**` | Keep `mod.zig`/`stub.zig` parity, flag-conditional builds | Parity fixes, passing toggle builds |
| **A2 Core Runtime** | `src/core/**`, `src/services/**` | Runtime/config contracts, integration boundaries | Stable runtime, focused tests |
| **A3 API and CLI** | `src/api/**`, CLI | Command behavior and help coherent with implementation | Passing CLI smoke, verified help |
| **A4 Validation** | Test and gate execution | Verification matrix, repro commands | Verification checklist and evidence |

### Plans index

- **Master:** [plans/plan.md](plans/plan.md) — Baseline, phases 0–3, constraints, definition of done.
- **Child plans:** See [CLAUDE.md — Skills, Plans, and Agents](CLAUDE.md#skills-plans-and-agents-full-index) for the full table. Plans in `plans/2026-02-*.md` often include a “For Claude: REQUIRED SUB-SKILL” line; follow phased gates and ownership when executing.
- **Quality / roadmap:** [docs/plan.md](docs/plan.md) (quality gates), [docs/roadmap.md](docs/roadmap.md) (milestones).

### Skills and rules

- **Skills:** `/baseline-sync`, `/zig-migrate` — see [CLAUDE.md — Custom skills](CLAUDE.md#custom-skills-invoke-by-name). Skill definitions: `.claude/skills/*/SKILL.md`.
- **Rules:** `.claude/rules/zig.md` — Zig 0.16 gotchas (auto-loaded for `.zig` files).
- **Ralph (separate):** `abi ralph run` / `abi ralph improve` — iterative agent with skill memory; not driven from this session unless the user asks.

### When to use which doc

| Goal | Use |
|------|-----|
| Build, test, gotchas, feature flags | [CLAUDE.md](CLAUDE.md) |
| Structure, coding style, testing, PRs | This file (AGENTS.md) |
| Full skills/plans/agents index | [CLAUDE.md — Skills, Plans, and Agents](CLAUDE.md#skills-plans-and-agents-full-index) |
| Execute a phased plan | [plans/plan.md](plans/plan.md) + specific child plan |
| Quality gates and baseline | [docs/plan.md](docs/plan.md), `scripts/project_baseline.env` |

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) — Workflow and PR checklist
- [CLAUDE.md](CLAUDE.md) — Build commands, gotchas, architecture; **[Working outside the Ralph loop](CLAUDE.md#working-outside-the-ralph-loop-cursor--claude)**; **[Skills, Plans, and Agents](CLAUDE.md#skills-plans-and-agents-full-index)**
- [SECURITY.md](SECURITY.md) — Security policy and reporting
