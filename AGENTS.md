# Repository Guidelines (AGENTS.md)

This document serves as the canonical reference for code agents and human contributors working within the ABI Framework. It defines the core commands, style expectations, and governance rules required to maintain codebase integrity.

## Governance & Onboarding

All contributors MUST read this document alongside `CONTRIBUTING.md` and `CLAUDE.md` before initiating changes. For automated agents, this file acts as the primary task-execution contract.

- **Consolidated Guidance**: Detailed FAQ and edge-case guidance are hosted in [docs/FAQ-agents.md](docs/FAQ-agents.md).
- **Consensus Rule**: Major architectural changes require tri-CLI consensus (where available) or explicit owner approval.
- **Rollout Phase**: We are currently in **Phase 4** of the framework rollout (see below).

## Core Commands

Use the Zig toolchain pinned in `.zigversion` (`0.16.0-dev.1503+738d2be9d`).

### Build & Lint
- `zig build`: Build the main framework and CLI.
- `zig build lint`: Check code formatting.
- `zig build fix`: Auto-format Zig sources.
- `./tools/scripts/fmt_repo.sh --check`: Repo-safe format check (skips vendored fixtures).

### Testing
- `zig build test --summary all`: Run the primary service test root.
- `zig build feature-tests --summary all`: Run manifest-driven feature coverage.
- `zig test <path> --test-filter "<pattern>"`: Run a single test or group.
- `zig build full-check`: Local confidence gate (format + tests + CLI checks).
- `zig build verify-all`: Release gate (full-check + examples + cross-compilation).

### Registry & Docs
- `zig build refresh-cli-registry`: Regenerate `tools/cli/generated/` after command changes.
- `zig build check-docs`: Verify documentation consistency.

## Coding Style & API Guidance

- **Formatting**: Rely exclusively on `zig fmt`. Never perform manual alignment.
- **Imports**: Use relative imports inside `src/features/`. Use `@import("abi")` for public consumers.
- **Naming**: `lower_snake_case` for functions/modules; `PascalCase` for types/structs.
- **Feature Gating**: Every feature module in `src/features/` must have a matching `mod.zig` and `stub.zig`. Public signatures must be identical in both.
- **Errors**: Use explicit error sets; propagate with `try`; never swallow errors silently.

Detailed style rules are maintained in the [Agent FAQ](docs/FAQ-agents.md#code-style--api-guidance).

## Agent Policy (Placeholders)

### Cursor Rules
Cursor-specific constraints and templates are located in [docs/guides/cursor_rules.md](docs/guides/cursor_rules.md). Direct Cursor rules (e.g., `.cursorrules`) are pending policy approval.

### Copilot Guidance
Copilot usage is permitted for boilerplate; however, all logic must be validated against the Zig 0.16 baseline. Annotate complex generated logic where appropriate.

## Commit & Pull Request Guidelines

- **Format**: Use short imperative subjects with prefixes: `fix:`, `feat:`, `docs:`, `chore:`, `style:`.
- **Scope**: Keep commits scoped to a single logical change or "wave".
- **Validation**: PRs must list the specific validation commands run (e.g., `full-check`) and their results.
- **Screenshots**: Required for TUI or dashboard UI changes.

## Phase 4 — Rollout Plan

1. **Validation & Stability (Current)**: Maintain compile-only macOS 26 bypasses; restore hosted CI gates.
2. **CEL Stage 0 Transition**: Shift default operations to `.zig-bootstrap/bin/zig`.
3. **Cleanup & Pruning**: Eliminate legacy `src/features/ai/personas/` after `profiles` migration.
4. **Documentation & Release**: Finalize `docs/api` generation and CLI command snapshots.

## Validation & Acceptance

A task is considered "Accepted" only when:
1. `zig build full-check` passes in a clean environment.
2. All touched modules' `stub.zig` files are verified against `mod.zig`.
3. The `tasks/todo.md` tracker is updated with completion evidence.
