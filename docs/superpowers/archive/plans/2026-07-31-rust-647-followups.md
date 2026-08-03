# Rust-rescoped follow-ups from archived Zig wave #647

> **Status:** Done — audit + MCP auth contracts landed with the
> `cursor/cleanup-refactor-finish` cleanup wave; the Rust-specific concurrency
> and benchmark guards closed in the follow-up hardening wave. This is not Zig
> plan execution.

## Context

[`docs/superpowers/archive/plans/2026-07-22-refactor-wave-647.md`](../archive/plans/2026-07-22-refactor-wave-647.md)
is Zig-era and must not be re-run. Useful themes, remapped to nightly Rust:

| Theme | Rust surface | Status |
| ----- | ------------ | ------ |
| Lock-across-I/O audit | `abi-wdbx` REST/cluster + rate limiter | Done (findings below) |
| MCP bad-token contracts | `abi-mcp` HTTP/SSE bearer | Done (`malformed_and_empty_bearer_schemes_are_unauthorized`) |
| Concurrency regression | DurableStore / REST lifecycle | Done (writer lock + 50-iteration tests) |
| Bench regression gate | `tools/` + `check.sh` | Done (same-host local guard) |
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

## Rust follow-up decisions (2026-07-31)

- `DurableStore` is explicitly single-writer per store path. An advisory lock
  held by the store session prevents two threads/processes from caching and
  appending the same next vector id; a second writer gets `WriterBusy`. The OS
  releases the lock when the owning file handle drops.
- The TCP lifecycle regression performs 50 complete query → joined server
  teardown → store drop → reopen/search cycles. It does not claim a concurrent
  use-after-drop test: Rust ownership makes that state unrepresentable.
- The historical Zig suite's 5% threshold was not copied onto the noisier Rust
  debug CLI workload. The live guard uses best-of-five p50 and a 25% threshold,
  after unchanged-code insertion p50 varied by roughly 28% locally. It compares
  only the committed OS/architecture class; unlike hosts report `SKIP` because
  cross-host timing deltas measure hardware, not a code regression.
- The benchmark output remains explicitly local and in-memory. Neither the
  baseline nor this gate is a published throughput/latency claim.

## MCP auth cases covered

- Missing `Authorization` when token configured → 401
- Wrong Bearer token (SSE) → 401
- Empty Bearer, `Basic` scheme, lowercase `bearer`, missing space → 401 without echoing the configured token
