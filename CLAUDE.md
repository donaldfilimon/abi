# CLAUDE.md — ABI

Zig 0.16 framework for AI services, vector search, and GPU compute. Pinned to `0.16.0-dev.2905+5d71e3051` (`.zigversion`).

## Commands

```bash
zig build test --summary all          # primary tests
zig build feature-tests --summary all # feature coverage
zig build full-check                  # pre-commit gate
zig build validate-flags              # flag combo check
zig build refresh-cli-registry        # after CLI changes
zig fmt --check build.zig build/ src/ tools/  # format check (always works)
```

**Darwin 25+**: `zig build` fails with linker errors. Use `./tools/scripts/run_build.sh <args>` instead. Never `use_lld = true` on macOS. Format checks (`zig fmt`) work without linking.

## Architecture

- `src/features/<name>/` — 19 comptime-gated modules, each with `mod.zig` + `stub.zig`
- `src/services/` — Connectors, LSP, MCP, runtime, security
- `src/core/` — Config, feature catalog, registry
- `build/` — Build system (options, flags, modules, test discovery)
- `tools/cli/` — CLI commands; registry in `tools/cli/registry/`

**mod/stub contract**: `stub.zig` must match `mod.zig` public signatures. Sub-module stubs not needed.

**Imports**: Use `@import("abi")` for framework API, relative imports within a feature. Never cross-directory relative imports.

## Conventions

- `zig fmt` only — never `zig fmt .` from root (walks vendored fixtures)
- `lower_snake_case` functions/files, `PascalCase` types
- Conventional commits, atomic scope
- Explicit error sets, propagate with `try`

## Zig 0.16 API Changes

- `std.time.unixSeconds()` not `timestamp()`
- `file.writeStreamingAll(io, data)` not `writeAll`
- `std.Io.Dir.createDirPath(.cwd(), io, path)` not `makeDirAbsolute`
- `.cwd_relative` / `.src_path` not `.path` on `LazyPath`
- `root_module` field not `root_source_file`
- `valueIterator()` not `.values()` on hash maps
- `@enumFromInt(x)` not `intToEnum`

## Feature Flags

All enabled by default. Disable: `-Dfeat-<name>=false`. GPU backend: `-Dgpu-backend=metal`.
25 flags in `build/options.zig`, 42 combos validated in `build/flags.zig`.

## Env Vars

`ABI_OPENAI_API_KEY`, `ABI_ANTHROPIC_API_KEY`, `ABI_OLLAMA_HOST`, `ABI_OLLAMA_MODEL`, `ABI_HF_API_TOKEN`, `DISCORD_BOT_TOKEN`

## Workflow

1. Review `tasks/lessons.md` at session start
2. Plan multi-file changes in `tasks/todo.md`
3. `zig build full-check` before completing (or `zig fmt --check` on Darwin)
4. Update `stub.zig` when changing `mod.zig` signatures
5. Update `tasks/lessons.md` after corrections
6. Version pin changes: update `.zigversion`, `build.zig.zon`, `baseline.zig`, `README.md`, CI config atomically

## References

- [AGENTS.md](AGENTS.md) — Workflow contract
- [tasks/lessons.md](tasks/lessons.md) — Correction log
