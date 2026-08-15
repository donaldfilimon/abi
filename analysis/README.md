# analysis/ — historical one-off analyses (index)

This directory held dated, Zig-era analysis documents produced during the
2026-07 reimagine/rewrite work. All of them described the **pre-Rust Zig tree**
(`src/*.zig`), which was fully replaced by the Rust workspace under `crates/`
(see `RUST-REWRITE-PLAN.md`). Consolidated 2026-08-14; this file is the
surviving index and record of conclusions.

## Where the files went

- `REIMAGINED_ARCHITECTURE-ZIG-ARCHIVE.md` (tracked) → moved to
  `docs/superpowers/archive/specs/`, alongside the other superseded Zig
  architecture specs.
- The untracked, gitignored working files — `abi/AI_NATIVE_SPEC.md`,
  `abi/BUSINESS_RULES.md`, `abi/DATA_OBJECTS.md`, `abi/IMPROVEMENT_PLAN.md`,
  `abi/MCP_INTEGRATION.md`, `src/AI_NATIVE_SPEC.md`, and
  `lock-io-audit-2026-07-22.md` — were parked outside the repo at
  `~/Archive/2026-08-14-md-cleanup/abi/analysis/` (this machine only; they were
  never committed).

## Dated conclusions preserved (only record)

- **Lock-across-network-I/O audit, 2026-07-22** (Zig tree): 23 lock
  acquisition sites reachable from the WDBX REST, MCP stdio/HTTP, and cluster
  RPC network paths were examined; **0 held a lock across socket I/O**. Every
  network write happened after the response was fully materialized in memory
  with all locks released (snapshot-copy pattern). Two notes worth carrying
  forward to any future audit of the Rust tree: (a) the Zig MCP ambient-store
  spinlocks were held across blocking *disk* I/O (safe by the audit's
  criterion, but a parking mutex or the snapshot pattern was recommended if
  MCP ever serves concurrent connections); (b) the cluster RPC node was
  lock-free by design under a single-threaded deterministic driver — a future
  "add a mutex" fix should guard only the apply calls, never the socket
  read/write.
- **Reimagine-era specs (2026-07-08 … 2026-07-16)**: the BUSINESS_RULES
  (~60 BR-* rules), DATA_OBJECTS, MCP_INTEGRATION, and AI_NATIVE_SPEC files
  reconstructed contracts from the Zig source. Their durable outcomes — the
  frozen 13-command CLI / 12-tool MCP surfaces, claim-honesty boundaries, and
  P0/drop decisions — live on in `docs/contracts/*.mdx`,
  `docs/spec/*.mdx`, `AGENTS.md`, and the golden fixtures under
  `tests/golden/`; the per-line Zig citations are obsolete.

New one-off analyses: prefer `docs/superpowers/` (plans/specs workflow) or a
dated file here, and record durable conclusions in `docs/` before archiving.
