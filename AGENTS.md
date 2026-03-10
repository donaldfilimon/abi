# Repository Guidelines

## Project Structure & Module Organization

`src/` holds the framework, with the public API rooted at `src/abi.zig`. Feature-gated modules live in `src/features/<name>/`; when you change a public feature API, update both `mod.zig` and `stub.zig`. Build logic is split across `build/`, CLI and TUI code live in `tools/cli/`, docs generation lives in `tools/gendocs/`, and CEL stage-0 sources live under `tools/cel/`, `stdlib/cel/`, and the root `./cel` launcher. Examples are in `examples/` and `examples/cel/`.

## Build, Test, and Development Commands

Use the Zig toolchain pinned in `.zigversion` (`0.16.0-dev.1503+738d2be9d`).

- `zig build`: build the main project.
- `zig build test --summary all`: run the primary test root in `src/services/tests/mod.zig`.
- `zig build feature-tests --summary all`: run manifest-driven feature coverage from `build/test_discovery.zig`.
- `zig build full-check`: local confidence gate for formatting, tests, docs, and CLI checks.
- `zig build verify-all`: broader release gate.
- `./tools/scripts/fmt_repo.sh [--check]`: repo-safe formatter surface; skips vendored bootstrap fixtures.
- `zig build fix` / `zig build lint`: format or format-check Zig sources.
- `zig build refresh-cli-registry`: regenerate `tools/cli/generated/` after command metadata changes.
- `./cel check`, `./cel run`, `./cel test`: exercise the CEL stage-0 toolchain.

## Coding Style & Naming Conventions

Let `zig fmt` set formatting; avoid manual alignment. Do not run `zig fmt .` from the repo root because `zig-bootstrap-emergency/` vendors intentionally invalid Zig fixtures. Use `./tools/scripts/fmt_repo.sh --check`, `zig build lint`, or `zig build fix`. Use relative imports inside `src/features/`, but use `@import("abi")` for public consumers. Keep file and function names in `lower_snake_case`. Feature flags use the `feat_<name>` pattern. CLI commands should keep their `pub const meta: command.Meta` block in sync with behavior and docs.

## Testing Guidelines

Write Zig tests with `test "..."` blocks close to the code they cover. Add broader feature coverage to `build/test_discovery.zig`, and keep CEL coverage under `tests/cel/`. For CLI changes, run `zig build cli-tests`, `zig build check-docs`, and refresh the CLI registry. Do not merge without `zig build full-check`, or record the exact environment blocker in your review notes.

## Commit & Pull Request Guidelines

Recent history favors short imperative subjects with prefixes such as `fix:`, `docs:`, `style:`, and `chore:`. Keep commits scoped to one change wave. PRs should summarize user-visible impact, list validation commands and results, link the relevant issue or task, and include screenshots only for TUI/dashboard changes. For non-trivial work, review `tasks/todo.md` and `tasks/lessons.md` first and capture outcomes in `tasks/todo.md`.
