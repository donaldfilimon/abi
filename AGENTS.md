# AGENTS.md

## Build Commands

**macOS 26.4+**: Use `./build.sh` (stock Zig LLD cannot link). **Linux/Older macOS**: Use `zig build` directly.

| Command                                              | Description                   |
| ---------------------------------------------------- | ----------------------------- |
| `./build.sh cli` / `zig build cli`                   | Build CLI binary              |
| `./build.sh check` / `zig build check`               | Lint + test + parity          |
| `./build.sh check-parity` / `zig build check-parity` | Verify mod/stub parity        |
| `zig build test -- --test-filter "pattern"`          | Run single test               |
| `zig build fix`                                      | Auto-format (NOT `zig fmt .`) |

## Critical Rules

1. **Never `@import("abi")` from `src/`** — causes circular import. Use relative imports.
2. **Mod/stub contract**: Every feature has `mod.zig` + `stub.zig`. Update both for any public API change.
3. After any public API change: run `zig build check-parity`.
4. Feature gates: `if (build_options.feat_X) @import("features/X/mod.zig") else @import("features/X/stub.zig")`.
5. **String ownership**: Use `allocator.dupe()` for string literals in structs with `deinit()`.

## Architecture

- `src/root.zig` — Package root, exports `abi.<domain>`
- `src/features/<name>/` — Feature modules with mod/stub/types pattern
- `src/main.zig` — CLI entry point
- `src/mcp_main.zig` — MCP server entry point

Feature flags (defaults enabled): `-Dfeat-gpu -Dfeat-ai -Dfeat-database -Dfeat-network -Dfeat-observability -Dfeat-web -Dfeat-cloud -Dfeat-auth -Dfeat-messaging -Dfeat-cache -Dfeat-storage -Dfeat-search`

GPU backends: `-Dgpu-backend=metal,cuda,vulkan,stdgpu`

## CLI Commands

```bash
abi                    # Smart status
abi version            # Version + build info
abi doctor             # Feature flags + GPU backends
abi features           # List 60 features
abi platform           # OS, arch, CPU, GPU
abi connectors         # 16 LLM providers + env vars
abi info              # Architecture summary
abi chat <msg>        # Multi-profile pipeline
abi db <cmd>          # Vector database
```

## Import Rules

- **Within `src/`**: `@import("../../foundation/mod.zig")` — never `@import("abi")`
- **From `test/`**: `@import("abi")` and `@import("build_options")` are wired by build.zig
- **Cross-feature**: Use conditional import pattern
- Explicit `.zig` extensions required

## Zig 0.17 Gotchas

- `ArrayListUnmanaged` init: `.empty` not `.{}`
- `std.BoundedArray` removed: use `buffer: [N]T = undefined + len: usize = 0`
- `std.time.milliTimestamp` removed: use `foundation.time.unixMs()`
- Entry: `pub fn main(init: std.process.Init) !void`
- `std.mem.trimRight` → `std.mem.trimEnd`
- Env vars: `std.c.getenv(name.ptr)` returns `?[*:0]const u8`

## Error Handling

- `@compileError` — Compile-time only
- `@panic` — Unrecoverable; CLI entry points and tests only
- `unreachable` — Provably impossible
- Error unions — Runtime failures in library code

## Known Pre-existing Test Failures

- Inference engine connector tests (2) — require external services
- Auth integration test (1) — requires ABI_JWT_SECRET env var
- Run `zig build test --summary all` to see pass/skip counts.

Glossary: See GLOSSARY.md for repo-wide terms.

Onboarding: See ONBOARDING.md for a quick-start onboarding guide.

## Where to start
- ONBOARDING.md for a quick bootstrap guide.
- GLOSSARY.md for glossary terms.
- CODEBASE_REVIEW.md for architecture and workflow guidance.

## Onboarding Checklist

- Would an agent likely miss this without help? Yes. Read ONBOARDING.md for a one-page onboarding guide.
- Would an agent likely miss this without help? Yes. Build CLI: `./build.sh cli` or `zig build cli`.
- Would an agent likely miss this without help? Yes. Build MCP: `./build.sh mcp` or `zig build mcp`.
- Would an agent likely miss this without help? Yes. Run parity checks: `zig build check-parity` or `./build.sh check-parity`.
- Would an agent likely miss this without help? Yes. Run full checks: `zig build check` or `./build.sh check`.
- Would an agent likely miss this without help? Yes. Run focused tests: `zig build test --summary all -- -test-filter "<pattern>"`.
- Would an agent likely miss this without help? Yes. After any public API change: run `zig build check-parity`.
- Would an agent likely miss this without help? Yes. Refer to CODEBASE_REVIEW.md for architecture and workflow guidance.
