# AGENTS.md - Canonical Repo Workflow blueprint

This document defines the foundational expectations for all contributors—human and
automated—working in the ABI repository. It serves as the authoritative task-execution
contract, ensuring architectural consistency and technical integrity.

---

## 1. Governance & Onboarding

All contributors MUST review this document alongside `CONTRIBUTING.md` and `CLAUDE.md`.
For automated agents, this file acts as the primary mandate for all execution waves.

- **Consolidated Guidance**: Detailed FAQs, style edge cases, and expanded command
  documentation are hosted in [docs/FAQ-agents.md](docs/FAQ-agents.md).
- **Consensus Rule**: Significant architectural changes or public API modifications
  require tri-CLI consensus (where available) or explicit owner approval.
- **Rollout Phase**: We are currently in **Phase 4: Rollout & Consolidation**.

## 2. Core Commands

Use the Zig toolchain pinned in `.zigversion` (`0.16.0-dev.1503+738d2be9d`).

### Build & Maintenance
- `zig build`: Build the main framework and CLI artifacts.
- `zig build fix`: Run the repository-safe auto-formatter.
- `zig build lint`: Verify formatting without applying changes.
- `./tools/scripts/fmt_repo.sh --check`: Lint-check core sources (skips vendored fixtures).
- `zig build refresh-cli-registry`: Update the generated CLI command metadata.

### Validation & Testing
- `zig build test --summary all`: Execute the primary service test suite.
- `zig build feature-tests --summary all`: Execute manifest-driven feature coverage.
- `zig build full-check`: Run the local CI-equivalent confidence gate.
- `zig build verify-all`: Execute the full release validation suite.
- `zig test <path> --test-filter "<pattern>"`: Run targeted tests by path or pattern.

## 3. Coding Style & API Guidance

- **Formatting**: Rely strictly on `zig fmt`. Never use manual vertical alignment.
- **Imports**: Use relative imports (`@import("local.zig")`) within feature modules.
  Use the canonical `@import("abi")` for all public framework consumption.
- **Naming**:
  - `lower_snake_case` for files, modules, and functions.
  - `PascalCase` for types, structs, and error sets.
- **Feature Gating**: Every feature module in `src/features/` must maintain a
  `mod.zig` (implementation) and a `stub.zig` (public signature mirror). Both
  files MUST have identical public signatures at all times.
- **Errors**: Use explicit error sets; propagate with `try`; avoid silent swallows.

## 4. Agent Policy (Placeholders)

### Cursor Rules
Cursor-specific constraints and task templates are defined in
[docs/guides/cursor_rules.md](docs/guides/cursor_rules.md). Direct Cursor rules
(e.g., `.cursorrules`) are pending formal policy approval.

### Copilot Guidance
Copilot usage is permitted for boilerplate. All logic MUST be manually validated
against the Zig 0.16 baseline and repo conventions. Annotate generated logic
where appropriate to indicate AI-assisted provenance.

## 5. Rollout Plan (Phase 4)

1. **Validation & Stability**: Maintain macOS 26+ bypass handling; restore CI gates.
2. **Migration**: Complete the `profiles` API transition and delete legacy `personas/`.
3. **Consolidation**: Prune duplicative guidelines across all root documentation.
4. **Release**: Finalize `docs/api` generation rules and CLI command snapshots.

## 6. Commit & Pull Request Guidelines

- **Format**: Use short imperative subjects with prefixes (e.g., `fix:`, `feat:`,
  `docs:`, `chore:`, `style:`).
- **Scope**: Keep commits atomic and scoped to a single logical change wave.
- **Validation trail**: Every PR description MUST include the summary of the
  `zig build full-check` command results from the target environment.

## 7. Acceptance Criteria

A task is considered complete and "Accepted" only when:
1. `zig build full-check` passes in a clean environment.
2. All touched modules' `stub.zig` files are verified against their `mod.zig` counterparts.
3. The `tasks/todo.md` tracker is updated with timestamped completion evidence.
4. Duplicative content has been successfully migrated to the [Agent FAQ](docs/FAQ-agents.md).

---

## Appendices & References

- [CONTRIBUTING.md](CONTRIBUTING.md): General contributor onboarding.
- [CLAUDE.md](CLAUDE.md): Quick reference for Claude Code sessions.
- [tasks/cleanup.md](tasks/cleanup.md): Active plan for documentation pruning.
- [docs/ZIG_MACOS_LINKER_RESEARCH.md](docs/ZIG_MACOS_LINKER_RESEARCH.md): Linker bypass logic.
