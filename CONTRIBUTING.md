# Contributing to ABI

## Quick start
- Install Zig (same version as CI).
- Clone & test:
  ```bash
  git clone https://github.com/donaldfilimon/abi.git && cd abi
  zig build test
  ```
- Format: `zig fmt .`

## Style
- 4-space indent; snake_case `.zig` filenames; camelCase for funcs.
- Public APIs have `///` doc comments.

## Tests
- Per-feature unit tests in `src/features/<name>/tests/mod.zig`.
- Integration tests in `tests/integration/` (use mock connector).
- CI must pass fmt + build + tests.

## Commits & PRs
- Branch: `feat/...`, `fix/...`, `refactor/...`.
- Message: short summary (<=72 chars), then details.
- PR checklist: fmt ✓, tests ✓, docs ✓, benchmarks (if perf-critical) ✓.

## Security
- No secrets in code or logs. Use env/CI secrets.

## Performance
- Include before/after benchmarks for perf-sensitive changes.
