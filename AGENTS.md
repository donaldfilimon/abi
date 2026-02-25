# Repository Guidelines

## Workflow Orchestration (Required)
- For non-trivial tasks (3+ steps, architectural decisions, or cross-file behavioral changes), write a checkable plan in `tasks/todo.md` before implementation.
- Define objective, scope, and verification criteria in `tasks/todo.md` before editing.
- Mark checklist items complete only after evidence exists.
- Stop and re-plan immediately if assumptions fail, commands block, or verification fails.
- Never mark work complete without proof (tests, logs, or behavior checks as appropriate).
- For UI changes, include visual verification; for API changes, validate the endpoint behavior.
- After any user correction, append a root-cause lesson and prevention rule to `tasks/lessons.md`.
- Review relevant lessons before starting related tasks.
- If using a reusable workflow contract, prefer `.codex/skills/workflow-orchestration/` as canonical and sync to `/Users/donaldfilimon/.codex/skills/workflow-orchestration/`.

## Project Structure & Module Organization
- `src/abi.zig`: public API entry point and compile-time feature wiring.
- `src/core/`: framework lifecycle, config, registry, and core errors.
- `src/features/<name>/`: feature implementation in `mod.zig` plus disabled `stub.zig` (public signatures must match).
- `src/services/`: always-available runtime/platform/connectors/shared utilities.
- `src/services/tests/mod.zig`: main integration/system test root.
- `src/feature_test_root.zig`: feature inline test root.
- `build/` and `build.zig`: modular build system and top-level build entry.
- `tools/cli/`: CLI commands/specs; `tools/scripts/`: quality-gate scripts.
- `examples/`: runnable examples for public API usage.

## Build, Test, and Development Commands
- `zig build`: build project defaults.
- `zig build run -- --help`: run CLI help and inspect command surface.
- `zig fmt .`: format all Zig code.
- `zig build lint`: CI-style formatting check (no writes).
- `zig build test --summary all`: run main test suite.
- `zig build feature-tests --summary all`: run feature inline tests.
- `zig build validate-flags`: compile-check feature-flag combinations.
- `zig build full-check`: pre-PR gate (format, unit tests, CLI smoke tests, flag validation).
- `zig build verify-all`: release-oriented gate (expands full-check with consistency/tests/examples/WASM checks).

## Coding Style & Naming Conventions
- Zig `0.16.0-dev` only (see `.zigversion`).
- 4 spaces, no tabs; keep lines near 100 chars.
- Naming: `PascalCase` (types/enums), `camelCase` (functions/vars), `snake_case.zig` (files).
- Use explicit imports only; never `usingnamespace`.
- Prefer specific error sets and `errdefer` for cleanup on error paths.
- Use `std.log.*` in library code; reserve `std.debug.print` for CLI/TUI output.

## Testing Guidelines
- Add or update tests with behavioral changes.
- Run focused checks during iteration: `zig test src/path/to/file.zig` or `--test-filter "pattern"`.
- For feature-module edits, update both `mod.zig` and `stub.zig`, then test both states:
  `zig build -Denable-<feature>=true` and `zig build -Denable-<feature>=false`.

## Commit & Pull Request Guidelines
<<<<<<< Current (Your changes)
- Follow Conventional Commit prefixes seen in history: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`.
- Keep commits scoped; avoid mixing refactors with behavior changes.
- PRs should include: clear summary, linked issue (if applicable), docs/example updates for API changes, and passing `zig build full-check`.

## Security & Configuration Tips
- Do not commit secrets, API keys, `.env`, `.zig-cache/`, or `zig-out/`.
- Report vulnerabilities through `SECURITY.md`.

## Canonical Sources
- Zig toolchain pin: `.zigversion`
- Command surface and build steps: `zig build --help`
- Test baseline source of truth: `tools/scripts/baseline.zig`
=======
- Commit messages follow Conventional Commits with optional scopes, e.g. `feat(metal): add unified memory` or `docs: update README`.
- Common types in history: `feat`, `fix`, `docs`, `chore`, `ci`, `security`, `refactor`, `test`.
- PRs should include a clear description, linked issues (if any), passing tests, and formatted code. Add benchmarks for performance-sensitive changes.

## References
- Architecture and contribution details: `docs/content/architecture.html`, `CONTRIBUTING.md`.
- Security and deployment guidance: `SECURITY.md`, `DEPLOYMENT_GUIDE.md`.
- Agent-specific requirements: `PROMPT.md` and `CLAUDE.md`.
>>>>>>> Incoming (Background Agent changes)
