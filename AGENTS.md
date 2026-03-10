# AGENTS.md - Canonical Workflow for Code Agents

This document defines the repo-wide expectations for automated and human agents
working in the ABI codebase. It complements CONTRIBUTING.md, CLAUDE.md, and the docs,
and serves as the contract used by agentic tools during plan-to-build work.
Last updated: 2026-03-10

## 1. Project Organization & Structure

The ABI project follows a modular, feature-oriented structure designed for high
concurrency and strict API gating.

- **`src/`**: Core framework logic.
  - **`abi.zig`**: Public API entry point.
  - **`features/`**: Feature-gated modules. Every feature MUST have a `mod.zig`
    (implementation) and a `stub.zig` (public signature mirror for gating).
- **`build/`**: Build system logic, target definitions, and test discovery.
- **`tools/`**: Internal tooling.
  - **`cli/`**: Main ABI CLI implementation.
  - **`scripts/`**: Maintenance and validation scripts.
- **`docs/`**: API documentation, architecture guides, and FAQs.
- **`examples/`**: Usage demonstrations and integration tests.

## 2. Core Development Commands

Use the Zig toolchain pinned in `.zigversion`.

### Build & Maintenance
- `zig build`: Build the framework and CLI artifacts.
- `zig build fix`: Run the repository-safe auto-formatter.
- `zig build lint`: Verify formatting without applying changes.
- `./tools/scripts/fmt_repo.sh --check`: Lint-check core sources (skips vendored fixtures).
- `zig build refresh-cli-registry`: Update the generated CLI command registry.

### Validation & Testing
- `zig build test --summary all`: Execute the primary service test suite.
- `zig build feature-tests --summary all`: Execute manifest-driven feature tests.
- `zig build full-check`: Run the local CI-equivalent confidence gate.
- `zig build verify-all`: Execute the full release validation suite.
- `zig test <path> --test-filter "<pattern>"`: Run targeted tests.

## 3. Coding Style & Conventions

- **Formatting**: Rely strictly on `zig fmt`. Never use manual vertical alignment.
- **Imports**: Use relative imports (`@import("local_file.zig")`) within feature
  modules. Use the canonical `@import("abi")` for all public framework consumption.
- **Naming**:
  - `lower_snake_case` for files, modules, and functions.
  - `PascalCase` for types, structs, and error sets.
  - `CONSTANT_CASE` for global constants and comptime values.
- **Errors**: Return explicit error sets; propagate using `try`; never swallow
  errors with `_ = ...` unless strictly justified and commented.
- **Feature Gating**: Public signatures in `mod.zig` and `stub.zig` MUST remain
  identical. Use the `is_blocked_darwin` flag to bypass linker issues on macOS 26+.

## 4. Agent Execution Policy

### Cursor & Copilot Guidelines
- **Cursor Rules**: Refer to `docs/guides/cursor_rules.md` for task templates and
  constraint definitions. `.cursorrules` are pending formal policy approval.
- **Copilot**: Usage is permitted for boilerplate generation. All generated
  logic MUST be manually validated against the Zig 0.16 baseline.

### Automated Refactoring
- Before performing batch edits, generate a research report or plan.
- Perform surgical `replace` calls instead of full-file rewrites where possible.
- Always run `full-check` after any automated modification wave.

## 5. Governance & Rollout (Phase 4)

We are currently in **Phase 4: Rollout & Consolidation**.

1. **Stability**: Maintain macOS 26 bypasses until upstream toolchain fixes land.
2. **Migration**: Move legacy `personas` to the new `profiles` API.
3. **Consolidation**: Centralize duplicative guidance from `AGENTS.md` and
   `CONTRIBUTING.md` into the [docs/FAQ-agents.md](docs/FAQ-agents.md).

## 6. Pull Request & Commit Standards

- **Commit Format**: Use short imperative subjects with prefixes (e.g., `fix:`,
  `feat:`, `docs:`, `chore:`).
- **Atomic Patches**: Each commit should represent a single logical change wave.
- **Validation**: Every PR description MUST include the output summary of the
  `zig build full-check` command.

## 7. Acceptance Criteria

A task is considered complete when:
- `zig build full-check` passes on the target platform.
- The `tasks/todo.md` entry is updated with a timestamped completion note.
- All related `stub.zig` files reflect the updated public signatures.
- Duplicative guidance has been relocated to the central FAQ.

---
*Refer to [docs/FAQ-agents.md](docs/FAQ-agents.md) for detailed style edge cases and
extended command documentation.*
