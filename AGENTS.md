# AGENTS.md

Zig 0.17.x/dev framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Entry Points

| Target | Build Command | Binary | Source |
|--------|--------------|--------|--------|
| CLI | `./build.sh cli` (macOS 26.4+) or `zig build cli` (Linux) | `zig-out/bin/abi` | `src/main.zig` |
| MCP server | `./build.sh mcp` | `zig-out/bin/abi-mcp` | `src/mcp_main.zig` |

## Build Commands

**macOS 26.4+ (Darwin 25.x)**: Always use `./build.sh`, never `zig build` directly — stock Zig's LLD cannot link. The wrapper auto-relinks with Apple's native linker.

**Linux / older macOS**: Use `zig build` directly.

| Command | Description |
|---------|-------------|
| `./build.sh` / `zig build` | Build static library |
| `./build.sh test --summary all` / `zig build test --summary all` | Run all tests |
| `zig build test -- --test-filter "pattern"` | Run single test |
| `./build.sh check` / `zig build check` | Lint + test + stub parity |
| `zig build check-parity` | Verify mod/stub declaration parity |
| `zig build fix` | Auto-format |
| `zig build cli` | Build CLI binary |
| `zig build mcp` | Build MCP server |
| `zig build feature-tests` | Run feature integration + parity tests |
| `zig build full-check` | Full validation gate |

### Focused Test Lanes (27 total)

Run unit + integration tests for specific features:

```bash
zig build {messaging,secrets,pitr,agents,multi-agent,orchestration,gateway,inference,gpu,network,web,observability,search,auth,storage,cloud,cache,database,connectors,lsp,acp,ha,tasks,documents,compute,desktop,pipeline}-tests
```

## Critical Rules

1. **Never `@import("abi")` from `src/`** — causes circular import. Use relative imports only.
2. **macOS 26.4+**: Use `./build.sh`, never `zig build` directly.
3. **Mod/stub contract**: Every feature has `mod.zig` (real), `stub.zig` (no-ops), `types.zig` (shared). Update both `mod.zig` and `stub.zig` together for any public API change.
4. **After any public API change**: Run `zig build check-parity` before committing.
5. **Feature gates**: Use pattern `if (build_options.feat_X) @import("features/X/mod.zig") else @import("features/X/stub.zig")`.
6. **String ownership**: Use `allocator.dupe()` for string literals in structs with `deinit()`.
7. **Imports**: Explicit `.zig` extension required on all path imports.

## Architecture

All enabled by default except `feat-mobile` and `feat-tui`:

```bash
zig build -Dfeat-gpu=false -Dfeat-ai=false
zig build -Dgpu-backend=metal
zig build -Dgpu-backend=cuda,vulkan
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

- 2 inference engine connector tests (expected failures)
- 1 auth integration test (expected failure)

## Toolchain

- **Zig version**: Pinned in `.zigversion` (currently `0.16.0`)
- **Zig manager**: `tools/zigly` — prefers `~/.zvm/bin/zig` when version matches

## See Also

- `CLAUDE.md` — Detailed architecture, conventions, and agent/skill references
- `QWEN.md` — Quick reference and Zig 0.16 gotchas

AiOps Adapter Refactor Plan (Centralized Pointer Cast Helper)
- Objective: Introduce a centralized single-argument pointer cast helper for the AiOps adapter to reduce duplication and improve consistency when casting from opaque pointers back to concrete Impl types.
- Scope: src/features/gpu/ai_ops/adapters.zig; ensure all internal adapter methods obtain Impl instances via the helper.
- Approach (phases):
  1) Add a small, centralized helper that converts *anyopaque to *Impl for the current AiOps Impl, taking the Impl as a comptime type parameter.
  2) Replace repetitive @ptrCast usages in AiOps adapter methods with calls to the centralized helper.
  3) Run a full build parity check: zig build check-parity. Address any parity or type-resolution issues.
  4) Run the test suite (zig build test or zig build test --summary all) if feasible in this repo context.
  5) Document the decision and usage pattern in AGENTS.md, including potential risks and how to extend to other adapters.
- Risks and caveats:
  - Cross-scope comptime type resolution can be tricky in Zig; ensure the helper is visible in the scope of all generically generated adapters.
  - Potential ABI/VTABLE compatibility concerns if the helper behavior is not perfectly aligned with existing casts; ensure parity checks pass.
- Verification plan:
  - Static checks: zig build check-parity; ensure no mod/stub parity regressions.
  - Unit tests: run any existing tests for AiOps paths; if none exist, add a minimal unit test to validate pointer-cast behavior using a mock Impl type.
- Contacts: If parity fails, revert changes and pursue a less invasive approach (e.g., a local inline cast helper per adapter invocation) to minimize risk.
