# Repository Guidelines

## Project Structure & Module Organization

ABI is a Zig 0.17 framework. Public exports start at `src/root.zig`; the CLI enters through `src/main.zig` and `src/abi_cli/`; the MCP server lives in `src/mcp/`. Feature code is under `src/features/<name>/` and each feature has both `mod.zig` and `stub.zig`. Connectors are in `src/connectors/`, shared runtime utilities in `src/core/` and `src/foundation/`, plugin manifests and examples in `src/plugins/`, and generated plugin metadata in `src/plugin_registry.zig`. Contract tests are in `tests/contracts/`; module and integration tests are mostly inline Zig `test {}` blocks plus `src/integration_tests.zig`.

## Build, Test, and Development Commands

Prefer `./build.sh` on macOS because it wraps the project Zig build flow.

```bash
./build.sh check              # primary gate: build, tests, fmt, parity, feature stubs
./build.sh full-check         # check + integration tests + benchmarks + TUI smoke
./build.sh cli                # builds zig-out/bin/abi
./build.sh mcp                # builds zig-out/bin/abi-mcp
zig build test                # module and connector tests
zig build test-integration    # integration suite
zig build benchmarks          # benchmark suite
zig build lint | zig build fix
zig build test -- --test-filter "<pattern>"
```

## Coding Style & Naming Conventions

Use `zig fmt` through `zig build lint` or `zig build fix`. Inside `src/`, use relative `.zig` imports only; the exceptions are `src/mcp/main.zig` and `src/mcp/handlers.zig`, which may import `abi`. Prefer Zig 0.17 patterns: `ArrayListUnmanaged(T).empty`, `std.mem.trimEnd`, `splitScalar`/`splitAny`/`splitSequence`, and `foundation.time.unixMs()`. Use `camelCase` functions/variables, `PascalCase` types, `SCREAMING_SNAKE_CASE` constants, and `snake_case` enum variants. Do not hand-edit `src/plugin_registry.zig`.

## Testing Guidelines

Run the narrowest meaningful test first, then `./build.sh check` before finishing. Public feature API changes require matching updates to `mod.zig` and `stub.zig`, then `zig build check-parity`. Disabled feature paths should return `error.FeatureDisabled`. Every module should include inline tests and `std.testing.refAllDecls(@This())`.

## Commit & Pull Request Guidelines

Recent history uses short scoped subjects such as `feat(wdbx): ...`, `docs(claude): ...`, and `fix: ...`. Keep commits focused; avoid `wip` commits in PR-ready branches. PRs should summarize behavior changes, list validation commands run, link related issues or specs, and call out CLI/MCP contract changes.

## Security & Configuration Tips

Trust executable config over prose when conflicts arise. Live connectors must require explicit credentials and transport selection; deterministic local helpers should not hit the network. Do not claim unverified capabilities such as distributed sharding, AES/RBAC, regulatory certifications, or benchmark numbers unless source, tests, or documented artifacts prove them.
