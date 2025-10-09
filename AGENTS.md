Repository Guidelines
=====================

This file is a living reference that contributors should read before making changes.  It reflects the
current layout and tooling of the ABI project and is intentionally concise (≈250 words).

## Project Structure & Module Organization

```
src/
├── main.zig               # Legacy bootstrap CLI
├── comprehensive_cli.zig  # Modern typed CLI (subcommands)
├── framework/             # Runtime core, config, registry
├── features/              # Feature namespaces (ai, database, gpu, web, monitoring, connectors)
├── shared/                # Logging, utilities, core helpers
└── core/                  # Low‑level containers & cleanup helpers

tests/                       # Mirrors src/ tree – *_*_test.zig files
benchmarks/                  # Performance tests & golden output
docs/                        # Markdown reference & examples
```
All public APIs live behind the re‑export in `src/mod.zig`; keep external imports unchanged to preserve compatibility.

## Build, Test, and Development Commands

```sh
zig build -Doptimize=ReleaseSafe
zig build test
zig fmt --check
zig build bench
zig build docs
```
All commands run from the repository root.

## Coding Style & Naming Conventions

* Indent with 2 spaces – `zig fmt` enforces this.
* Public identifiers use `snake_case`; types/enums use `CamelCase`.
* Allocate via the callerʼs allocator; never use the global allocator.
* Use tagged errors (`error{...}`).
* Do not modify `src/root.zig` unless adding backward‑compatibility shims.
Run `zig fmt` after edits.

## Testing Guidelines

* Tests mirror src/ and end with `_test.zig`.
* Use `std.testing.refAllDecls` to guard against API drift.
* Database tests cover ID monotonicity, recall@k, and concurrency (≥8 writers/32 readers).
* Run tests with `zig build test`; CI enforces 90 %+ branch coverage.

## Commit & Pull Request Guidelines

* Conventional Commits: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`.
* Commit messages must start with a concise subject (≤72 chars).
* PRs need an actionable title, description, linked issue(s), and test or benchmark diff.
* All PRs must pass CI (build, format, test, benchmark, security).
* Major refactors should add backward‑compatibility shims in `src/root.zig`.
* When changing public APIs, add a deprecation comment and bump the major part of the version in `CHANGELOG.md`.

## Security & Configuration Tips

* Enable optional back‑ends with `-Denable-gpu` / `-Denable-web`.
* Store confidential config outside the repo; read via environment variables.
* Run `zig build test` / `zig build bench` with network access restricted for local dev;
  CI has full network access.

---
For additional questions, open an issue or reach out on the project Slack.
