# Rust-rescoped follow-ups from archived Zig wave #647

> **Status:** Partial — audit + MCP auth contracts landed with the
> `cursor/cleanup-refactor-finish` cleanup wave. Remaining items stay on
> `tasks/todo.md` as optional hardening, not Zig plan execution.

## Context

[`docs/superpowers/archive/plans/2026-07-22-refactor-wave-647.md`](../archive/plans/2026-07-22-refactor-wave-647.md)
is Zig-era and must not be re-run. Useful themes, remapped to nightly Rust:

| Theme | Rust surface | Status |
| ----- | ------------ | ------ |
| Lock-across-I/O audit | `abi-wdbx` REST/cluster + rate limiter | Done (findings below) |
| MCP bad-token contracts | `abi-mcp` HTTP/SSE bearer | Done (`malformed_and_empty_bearer_schemes_are_unauthorized`) |
| Concurrency regression | DurableStore / REST vs deinit | Optional follow-up |
| Bench regression gate | `tools/` + `check.sh` | Optional follow-up |
| HA/ACP stubs | N/A (no Rust feature flags) | Won't port as Zig `feat_ha` |

## Lock audit findings (2026-07-31)

| Site | Lock | Held across I/O? | Verdict |
| ---- | ---- | ---------------- | ------- |
| `crates/abi-wdbx/src/rate_limit.rs` | `Mutex<State>` | No — only refill/acquire math + stats copy | OK |
| `crates/abi-wdbx/src/mvcc.rs` | `RwLock<ChainState>` | No network I/O; in-process chain ops | OK for single-host demo |
| `crates/abi-wdbx/src/rest.rs` | (none own) | Uses `RateLimiter::acquire` then handles request | OK |
| `crates/abi-wdbx/src/cluster_rpc.rs` | (none) | Single-threaded accept loop | OK |
| `crates/abi-mcp/src/http.rs` | `AtomicBool` stop only | Auth check before body handling | OK |

No lock-across-`TcpStream` read/write found on the loopback REST/cluster/MCP paths.

## MCP auth cases covered

- Missing `Authorization` when token configured → 401
- Wrong Bearer token (SSE) → 401
- Empty Bearer, `Basic` scheme, lowercase `bearer`, missing space → 401 without echoing the configured token
