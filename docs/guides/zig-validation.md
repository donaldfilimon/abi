# Zig Validation Guide

Single source of truth for Zig toolchain setup, build commands, and API compatibility for this codebase.

## Toolchain Pin

This project is pinned to Zig `0.16.0-dev.2934+47d2e5de9` (see `.zigversion`). If your Zig version doesn't match, `build.zig` detects the mismatch and prints clear instructions.

## Build Commands

```bash
zig build                             # build all targets
zig build run -- --help               # run the CLI
zig build test --summary all          # primary tests
zig build feature-tests --summary all # feature coverage
zig build full-check                  # pre-commit gate
zig build validate-flags              # flag combo check
zig build check-stub-parity           # mod/stub declaration parity
zig build toolchain-doctor            # inspect active Zig resolution
zig build check-zig-version           # verify pin + docs consistency
zig build preflight                   # integration environment diagnostics
zig build refresh-cli-registry        # after CLI changes
zig build gendocs                     # regenerate docs
zig build check-docs                  # verify docs consistency
zig build cli-tests                   # CLI smoke tests (~53 vectors)
zig build cli-tests-full              # exhaustive CLI integration tests
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/  # format check (always works)
```

**Running a single test**: `zig test src/path/to/file.zig -fno-emit-bin` for standalone files. For module-integrated tests, use `zig build test --summary all` with feature flags to narrow scope (e.g., `zig build test -Dfeat-ai=false -Dfeat-gpu=false --summary all` to skip AI and GPU tests).

## Darwin 25+ / macOS 26+

Stock prebuilt Zig's internal LLD linker fails with undefined symbols (`_malloc_size`, etc.) at link time on Darwin 25+. Workarounds:

- Use a pinned Zig matching `.zigversion` on PATH for full gate coverage
- Never set `use_lld = true` on macOS (zero Mach-O support)
- Format checks (`zig fmt --check ...`) always work as a fallback gate

## Verification Gates

Run the strongest gate your environment supports:

| Gate | Command | When |
|------|---------|------|
| Format check | `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` | Every change (always works) |
| Full check | `zig build full-check` | Before completing (requires pinned Zig matching `.zigversion` on PATH) |
| Darwin fallback | `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/` | When stock Zig is linker-blocked on Darwin 25+ |
| Full release | `zig build verify-all` | Release prep |

## Zig 0.16 API Changes

Migration notes from 0.15 and earlier:

- `std.heap.DebugAllocator(.{}){}` not `GeneralPurposeAllocator`
- `std.time.unixSeconds()` not `timestamp()`
- `file.writeStreamingAll(io, data)` not `writeAll`
- `std.Io.Dir.createDirPath(.cwd(), io, path)` not `makeDirAbsolute`
- `.cwd_relative` / `.src_path` not `.path` on `LazyPath`
- `root_module` field not `root_source_file`
- `valueIterator()` not `.values()` on hash maps
- `@enumFromInt(x)` not `intToEnum`
- `ArrayListUnmanaged` / `AutoHashMapUnmanaged` init: `.empty` not `.{}`
- `pub fn main(init: std.process.Init) !void` not `pub fn main() !void`
- `std.io.fixedBufferStream` does not exist — use manual buffer slicing
- `std.time.Instant` does not exist — use `std.c.clock_gettime(.MONOTONIC, &ts)`
- `_ = param` after referencing `param` triggers "pointless discard" — use `_:` prefix in signature
- `defer` on lists returned via `toOwnedSlice()` causes use-after-free — use `errdefer` when returning owned memory

## CLI Quick Reference

```bash
abi --help                # top-level help
abi system-info           # runtime / diagnostics
abi doctor                # health check
abi db stats              # database stats
abi db query --embed "q" --top-k 5  # vector search
abi agent                 # AI agent
abi gpu summary           # GPU info
abi gendocs --check       # docs consistency
```

CLI commands live in `tools/cli/commands/`. After adding/changing commands, run `zig build refresh-cli-registry`.

## Benchmarks

```bash
zig build benchmarks                       # Run all suites
zig build benchmarks -- --suite=simd       # Run specific suite
zig build benchmarks -- --quick            # Fast CI-friendly run
zig build bench-competitive                # Industry comparisons
```

Suites cover SIMD, memory, concurrency, database, network, crypto, AI, and GPU workloads. See `benchmarks/README.md` for details.

## Feature Flags

All enabled by default (except `feat-mobile`, which defaults to `false`). Disable: `-Dfeat-<name>=false`. GPU backend: `-Dgpu-backend=metal`.
27 flags in `build/options.zig`, 58 combos validated in `build/flags.zig`.
Catalog source of truth: `src/core/feature_catalog.zig`.

## Env Vars

`ABI_OPENAI_API_KEY`, `ABI_ANTHROPIC_API_KEY`, `ABI_OLLAMA_HOST`, `ABI_OLLAMA_MODEL`, `ABI_HF_API_TOKEN`, `DISCORD_BOT_TOKEN`
