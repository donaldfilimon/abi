# FAQ: Centralized Agent Guidance

Overview
- This document consolidates guidance previously scattered across AGENTS.md and related docs.
- It is the canonical reference for build/test commands, code style, and governance expectations.

Who is this for?
- Code agents (automated reviewers, plan executors) and human contributors who work in the ABI repo.

Core Commands & Guidance
- Build: `zig build`.
- Tests: `zig build test --summary all`.
- Feature tests: `zig build feature-tests --summary all`.
- Full checks: `zig build full-check`.
- Validation: `zig build verify-all`.
- Lint & format: `zig build lint`; `./tools/scripts/fmt_repo.sh --check`.
- Auto-format: `zig build fix`.
- CLI registry: `zig build refresh-cli-registry` and `zig build check-cli-registry`.
- Run a single test: `zig test <path> --test-filter "<pattern>"`.
- Run a single test by name: `zig test <path> --test-filter "<TestName>"`.
- Example: `zig test src/services/tests/mod.zig --test-filter "Network"`.

Code Style & API Guidance
- Imports: prefer relative within modules; public consumers import `abi` root.
- Formatting: rely on Zig tooling; avoid manual formatting tricks.
- Naming: snake_case for functions/modules; PascalCase for types; descriptive enums.
- Errors: explicit error sets; propagate with `try`; avoid silent swallow.
- Public API: mark with `pub`; mirror signatures in `stub.zig` for gating.
- Comments: keep concise docs for non-trivial logic; avoid noise.
- Concurrency: document synchronization; prefer scoped lifetimes.
- Observability: standardize logging; avoid stray prints in library code.
- Docs alignment: ensure docs reflect public surface and testing.

Testing Guidelines
- Write Zig tests with `test "..."` blocks close to the code they cover.
- Update the feature-test manifest in `build/test_discovery.zig` when adding new feature modules or files.
- CEL stage-0 coverage belongs under `tests/cel/`.
- For CLI changes, run `zig build cli-tests` and verify the generated registry.

Cursor Rules (Placeholder)
- Cursor guidance is not configured in this repo yet. When policy exists, place rules in `docs/guides/cursor_rules.md` and reference here.

Copilot Guidance (Placeholder)
- If Copilot usage occurs, annotate decisions and caveats inline and in this document.

Governance & Onboarding
- Read this document with CONTRIBUTING.md and CLAUDE.md before making changes.
- Before major changes, run `zig build full-check` and `zig build verify-all`.
- Plan multi-file changes; include target file paths and proposed diffs before execution.

Relocation & Centralization
- Duplicated material from AGENTS.md has been consolidated into this FAQ.
- See AGENTS.md for contract and summary; see this FAQ for reference content.

Validation & Acceptance
- Provide reproduction steps; show results.
- All builds/tests pass in clean environments; report blockers.
- If introducing new public usage, include a small usage example.
