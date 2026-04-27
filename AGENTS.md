# AGENTS.md

Zig 0.17.x/dev framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Build Commands

**macOS 26.4+**: Always use `./build.sh`, never `zig build` directly — stock Zig's LLD cannot link.
**Linux / older macOS**: Use `zig build` directly.

| Command | Description |
|---------|-------------|
| `./build.sh cli` / `zig build cli` | Build CLI binary (`zig-out/bin/abi`) |
| `./build.sh mcp` / `zig build mcp` | Build MCP server (`zig-out/bin/abi-mcp`) |
| `./build.sh test --summary all` | Run all tests |
| `zig build test -- --test-filter "pat"` | Run single test |
| `./build.sh check` | Lint + test + stub parity |
| `zig build full-check` | Full validation gate (macOS: `./build.sh full-check`) |
| `zig build check-parity` | Verify mod/stub declaration parity |
| `zig build fix` | Auto-format |
| `zig build cross-check` | Verify cross-compilation (linux/wasi) |
| `zig build typecheck` | Compile-only validation |

**Test lanes** (27 total): `zig build {messaging,secrets,pitr,agents,gpu,network,web,search,auth,storage,cloud,cache,database,connectors,lsp,acp,ha,tasks,documents,compute,desktop,pipeline}-tests`

## Critical Rules

1. **Never `@import("abi")` from `src/`** — causes circular import. Use relative imports only.
2. **Mod/stub contract**: Every feature has `mod.zig` (real), `stub.zig` (no-ops), `types.zig` (shared). Update both together for any public API change.
3. **After any public API change**: Run `zig build check-parity` before committing.
4. **Feature gates**: `if (build_options.feat_X) @import("features/X/mod.zig") else @import("features/X/stub.zig")`
5. **String ownership**: Use `allocator.dupe()` for string literals in structs with `deinit()`.

## Architecture

### Module Layout

- `src/root.zig` — Package root, re-exports all domains as `abi.<domain>`
- `src/core/` — Always-on internals: config, errors, registry, framework lifecycle
- `src/features/` — 21+ feature directories under mod/stub/types pattern
- `src/foundation/` — Shared utilities: logging, security, time, SIMD, sync primitives
- `src/runtime/` — Task scheduling, event loops, concurrency primitives
- `src/platform/` — OS detection, capabilities, environment abstraction
- `src/connectors/` — External service adapters (OpenAI, Anthropic, Discord)
- `src/protocols/` — MCP, LSP, ACP, HA protocol implementations
- `src/tasks/` — Task management, async job queues
- `src/inference/` — ML inference: engine, scheduler, sampler, KV cache

### The Mod/Stub Pattern

Every feature under `src/features/<name>/` follows a contract:
- `mod.zig` — Real implementation
- `stub.zig` — API-compatible no-ops (same public surface, zero-cost when disabled)
- `types.zig` — Shared types used by both mod and stub

In `src/root.zig`, each feature uses comptime selection:
```zig
pub const gpu = if (build_options.feat_gpu)
    @import("features/gpu/mod.zig")
else
    @import("features/gpu/stub.zig");
```

**Critical rule**: When modifying a feature's public API, **both `mod.zig` and `stub.zig` must be updated in sync**. Run `zig build check-parity` to verify.

## Import Rules

- **Within `src/`**: Use relative imports only (`@import("../../foundation/mod.zig")`). Never `@import("abi")` from inside the module — causes circular import error.
- **From `test/`**: Use `@import("abi")` and `@import("build_options")` — these are wired as named module imports by build.zig.
- **Cross-feature imports**: Never import another feature's `mod.zig` directly. Use conditional: `const obs = if (build_options.feat_observability) @import("../../features/observability/mod.zig") else @import("../../features/observability/stub.zig");`
- **Explicit `.zig` extensions** required on all path imports.

## Build Options

Feature flags (defaults to enabled): `-Dfeat-gpu=true -Dfeat-ai=true -Dfeat-database=true -Dfeat-network=true -Dfeat-observability=true -Dfeat-web=true -Dfeat-cloud=true -Dfeat-analytics=true -Dfeat-auth=true -Dfeat-messaging=true -Dfeat-cache=true -Dfeat-storage=true -Dfeat-search=true -Dfeat-mobile=false -Dfeat-tui=false`

GPU backends: `-Dgpu-backend=metal,cuda,vulkan,stdgpu` (stdgpu is CPU-based fallback)

## CLI Commands

```bash
abi                    # Smart status (feature count, enabled/disabled tags)
abi version            # Version and build info
abi doctor             # Build config report (all feature flags + GPU backends)
abi features           # List all 60 features with [+]/[-] status
abi platform           # Platform detection (OS, arch, CPU, GPU backends)
abi connectors         # List LLM provider connectors with env var status
abi search <sub>       # Full-text search
abi info               # Framework architecture summary
abi db <subcommand>    # Vector database (add, query, stats, serve)
abi chat <message...>  # Route through multi-profile pipeline
abi serve              # Start ACP HTTP server
abi dashboard          # Developer diagnostics shell (requires -Dfeat-tui=true)
abi help               # Full help reference
```

## Zig 0.17 Gotchas

- `ArrayListUnmanaged` init: use `.empty` not `.{}`
- `std.BoundedArray` removed: use manual `buffer: [N]T = undefined` + `len: usize = 0`
- `std.Thread.Mutex` may be unavailable: use `foundation.sync.Mutex`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- Entry points: `pub fn main(init: std.process.Init) !void` (not old `pub fn main() !void`)
- `std.mem.trimRight` renamed to `std.mem.trimEnd`
- Runtime env var access: `std.c.getenv(name.ptr)` returns `?[*:0]const u8`
- Signal handlers: use `std.posix.Sigaction` with `callconv(.c)` handler functions
- **Do NOT run `zig fmt .`** at repo root — use `zig build fix`

## Error Handling Convention

- `@compileError` — Compile-time contract violations only
- `@panic` — Unrecoverable invariant violations; only in CLI entry points and tests, never in library code
- `unreachable` — Provably impossible branches where compiler can verify exhaustiveness
- Error unions — All runtime failure paths in library code; prefer `error.FeatureDisabled` in stubs

## Testing

- `zig build test --summary all` — Run all unit + integration tests
- `zig build test -- --test-filter "pattern"` — Run single test by name
- Files end with: `test { std.testing.refAllDecls(@This()); }`

**Known pre-existing failures** (not regressions):
- Inference engine connector backend tests (2)
- Auth integration tests (1 failure, 3 leaks)

## Toolchain

- **Zig**: pinned in `.zigversion` (`0.17.0-dev.27+0dd99c37c`)
- **Bootstrap**: `tools/zigly --bootstrap` (one-command project setup)
- **Status**: `tools/zigly --status` (print pinned zig path)
- **Cross-compile check**: `zig build cross-check` validates linux/wasi targets

## MCP Server

- Build: `zig build mcp` → `zig-out/bin/abi-mcp`
- Install: `cp zig-out/bin/abi-mcp ~/.local/bin/`
- Config: `.mcp.json` (root)

## Key References

- **Full architecture**: See `README.md` and `docs/spec/ABBEY-SPEC.md`
- **Import rules and conventions**: See `CLAUDE.md` (comprehensive, updated Apr 2026)
- **Onboarding**: See `docs/onboarding.md`
- **AI pipeline spec**: See `docs/spec/abbey-aviva-abi-framework.md`

## Workflow Notes

- Run `zig build check-parity` after ANY public API change
- Verification gate before committing: `./build.sh check` (macOS) or `zig build check` (Linux)
- Files with pre-existing refAllDecls errors are documented in CLAUDE.md
- **No `rm`**: Use safe alternatives
