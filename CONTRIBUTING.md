# Contributing to ABI

## Quick start
- Install Zig (same version as CI).
- Clone & test:
  ```bash
  git clone https://github.com/donaldfilimon/abi.git && cd abi
  zig build test
  ```
- Format: `zig fmt .`

## Repository Structure (Post-Refactor)

The codebase has a clear separation between library and application code:

### `lib/` - Core Library (Primary)
- **All framework library code goes here**
- Reusable modules and features
- Single source of truth for library implementation
- Entry point: `lib/mod.zig`

### `src/` - Application Code
- CLI implementation
- Application-specific tools
- Examples and demos
- Integration tests

### Adding Code

**For library features:**
1. Add to `lib/` directory (e.g., `lib/features/myfeature/`)
2. Update appropriate `mod.zig` to export (e.g., `lib/features/mod.zig`)
3. Import via `@import("abi")` from application code
4. Add tests in `tests/` directory

**For application code:**
1. Add to `src/` or appropriate directory
2. Import library via `@import("abi")`
3. No library code should go in `src/`

## Style
- 2-space indent (enforced by `zig fmt`)
- `snake_case` for `.zig` filenames and public identifiers
- `CamelCase` for types and enums
- Public APIs have `///` doc comments

## Tests
- Unit tests mirror the `lib/` structure in `tests/unit/`
- Integration tests in `tests/integration/`
- Feature tests alongside feature code
- CI must pass fmt + build + tests

## Commits & PRs
- Branch: `feat/...`, `fix/...`, `refactor/...`
- Conventional Commits: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `chore:`
- Message: short summary (≤72 chars), then details
- PR checklist: 
  - [ ] `zig fmt` passes
  - [ ] `zig build test-all` passes
  - [ ] Documentation updated
  - [ ] Benchmarks included (if perf-critical)
  - [ ] Code added to correct directory (`lib/` for library, `src/` for app)

## Security
- No secrets in code or logs. Use env/CI secrets.

## Performance
- Include before/after benchmarks for perf-sensitive changes.
