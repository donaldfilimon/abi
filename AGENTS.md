# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the core Zig 0.16 implementation. Nested modules should import dependencies via their parent `mod.zig` (parent exports, child imports).
- Public API imports should use `@import("abi")` (avoid direct file paths).
- When changing a feature module's public API, update both `mod.zig` and `stub.zig`.
- `tests/` holds integration-style tests; unit tests also live alongside code as `*_test.zig`.
- `benchmarks/` contains performance suites; `examples/` has runnable samples.
- `docs/` and `tools/` host documentation and developer tooling.

## Build, Test, and Development Commands
- `zig build` — build the project.
- `zig build test --summary all` — run the full test suite (baseline around 787/792).
- `zig test src/path/to/file.zig --test-filter "pattern"` — run a focused test.
- `zig fmt .` — format all Zig files (required after edits).
- `zig build cli-tests` — run CLI command smoke tests.
- `zig build full-check` — format + tests + CLI smoke tests.
- `zig build bench-competitive` / `zig build benchmarks` — performance validation.
- `zig build run -- --help` — run the CLI entry point.

## Change Management
- Run `git status` and `git diff --stat` before edits to review existing work.
- Avoid reverting unrelated changes; ask if large or unclear diffs exist.
- Use package managers to add new dependencies (latest versions).

## Coding Style & Naming Conventions
- Indentation: 4 spaces, no tabs; keep lines under ~100 chars.
- Types: `PascalCase`; functions/variables: `camelCase`; errors: `*Error`; configs: `*Config`.
- Prefer explicit imports (avoid `usingnamespace`).
- Use specific error sets and `error!Type` returns; clean up with `defer`/`errdefer`.
- Prefer `std.ArrayListUnmanaged` and modern format specifiers (e.g., `{t}` for enums/errors).

## Testing Guidelines
- Use `std.testing` and `GeneralPurposeAllocator` in new tests.
- Place unit tests in `*_test.zig`; integration tests in `tests/`.
- For feature-gated tests, use `-Denable-gpu=true`; skip hardware-gated tests via `error.SkipZigTest`.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits with optional scopes, e.g. `feat(metal): add unified memory` or `docs: update README`.
- Common types in history: `feat`, `fix`, `docs`, `chore`, `ci`, `security`, `refactor`, `test`.
- PRs should include a clear description, linked issues (if any), passing tests, and formatted code. Add benchmarks for performance-sensitive changes.

## References
- Architecture and contribution details: `docs/content/architecture.html`, `CONTRIBUTING.md`.
- Security and deployment guidance: `SECURITY.md`, `DEPLOYMENT_GUIDE.md`.
- Agent-specific requirements: `PROMPT.md` and `CLAUDE.md`.
