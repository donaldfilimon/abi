# abi `src/` touchpoints for Zig master drift

When master breaks the driver gates, inspect these surfaces first. Trust
executable sources over this list if they diverge.

## Toolchain pin surfaces (not `src/`, but co-move)

| Path | Role |
|------|------|
| `.zigversion` | Live pin string |
| `build.zig.zon` `minimum_zig_version` | Lower bound (may lag pin) |
| `.github/workflows/ci.yml` `ZIG_VERSION` | CI pin; must match `.zigversion` when either moves |
| `build.zig` / `tools/build.sh` / `./build.sh` | Invoke PATH `zig`; no auto-switch |

## High-churn Zig 0.17 patterns (abi conventions)

From `AGENTS.md` / live code:

| Pattern | Where it bites |
|---------|----------------|
| `pub fn main(init: std.process.Init) !void` | `src/main.zig`, MCP entry |
| `ArrayListUnmanaged(T).empty` | Many modules; old `.init(allocator)` fails |
| `std.mem.trimEnd` (not `trimRight`) | String helpers |
| `splitScalar` / `splitAny` / `splitSequence` | Parsers |
| Relative `.zig` imports inside `src/` | Feature modules; only MCP handler group may `@import("abi")` |
| `foundation.time.unixMs()` | Timestamps |
| Explicit allocator; no global hidden allocator | New APIs |

## Network / IO (WDBX + MCP)

Zig 0.16 fails on WDBX/MCP listeners that use `std.Io.net.Stream`. Master
churn often hits:

| Area | Paths |
|------|--------|
| WDBX durable / REST | `src/features/wdbx/durable_store.zig`, `rest.zig`, cluster RPC |
| MCP transports | `src/mcp/server.zig`, HTTP/SSE listeners |
| Connectors HTTP | `src/connectors/http.zig` (SSE streaming) |

Linux ambient store note (cloud VM): `setPermissions`/`fchmod` BADF on some
nightlies — use `ABI_WDBX_PERSIST=0` for ambient paths; not a master-only issue
but confuses smoke on Linux.

## Build-feature graph

Feature flags select `mod.zig` vs `stub.zig` under `src/features/`. Master
breaks may appear only with default-on features:

- `feat-foundationmodels` — arm64 macOS + `xcrun swiftc` (skip with
  `-Dfeat-foundationmodels=false` when diagnosing pure Zig drift)
- GPU Metal link path on macOS via `./build.sh`

## Driver gate → likely blame

| Failing gate | Look first |
|--------------|------------|
| `check-parity` | Toolchain / build.zig options / parity tool itself |
| `./build.sh cli` | `src/main.zig`, `src/cli/**`, `src/features/**` graph |
| `./build.sh mcp` | `src/mcp/**`, shared AI/WDBX imports |
| `abi help` / `abi backends` | Runtime panic, missing dylib (FM shim), PATH |
| `run-abi smoke` | Env, WDBX persist, MCP stdio protocol |

## Fix discipline

1. Minimal source fix; no drive-by refactors.
2. Preserve mod/stub parity (`zig build check-parity`) if public AI/feature APIs change.
3. Re-run `zig-master-check.sh` then, if promoting a pin, full `./build.sh check` on the new pin.
4. Never force-push `main`; pin bumps use a normal PR.
