---
name: run-abi
description: Build, launch, and drive the abi nightly-Rust project — the `abi` CLI and the `abi-mcp` JSON-RPC server. Use when asked to run, start, build, smoke-test, or screenshot abi, drive its CLI subcommands, exercise the MCP server over stdio, or confirm a change works in the real binaries (not just the test suite).
---

# Run abi

`abi` is a **nightly Rust** framework that builds two binaries: a CLI
(`target/debug/abi`) and an MCP server (`target/debug/abi-mcp`, JSON-RPC 2.0 over
stdio + optional loopback HTTP/SSE). Both are **non-interactive and
headless-friendly** — the driver builds them and drives the real binaries
end-to-end.

**Paths below are relative to the repo root.** The driver lives at
`.agents/skills/run-abi/smoke.sh` and resolves the repo root from its own location.

## Run (agent path) — the driver

```bash
./.agents/skills/run-abi/smoke.sh
```

Expected tail on success (exit 0):

```
=== summary: pass=N fail=0 ===
transcript: <repo>/target/debug/run-abi-smoke.txt
SMOKE OK
```

## Prerequisites

- Nightly Rust via `rust-toolchain.toml`. **Always** use `./tools/cargo.sh`
  (never bare Homebrew `cargo`).
- Primary gate: `./tools/check.sh`.

## Build

```bash
./tools/cargo.sh build -p abi-cli   # -> target/debug/abi
./tools/cargo.sh build -p abi-mcp   # -> target/debug/abi-mcp
```

On arm64 macOS, building with the default `foundationmodels` feature also
produces `target/debug/libabi_fm_shim.dylib` next to the binaries.

## Drive the CLI directly

```bash
./target/debug/abi help
./target/debug/abi backends
./target/debug/abi scheduler status
./target/debug/abi complete "summarize scheduler status"
./target/debug/abi plugin list
```

## Drive the MCP server directly

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | ./target/debug/abi-mcp 2>/dev/null
```

Exactly **12** frozen tools. Prefer `mcp/launcher.sh` when `@loader_path` for
the FM dylib matters.

## Gotchas

- **GPU honesty:** `accelerated=false` means native kernels are not linked;
  CPU SIMD is the real path. Do not treat Metal-as-preferred as native kernels.
- **`complete` (no `--live`)** is fully local. `--live` is Anthropic-only for HTTP;
  `apple-fm --confirm` uses the FoundationModels shim when ready.
- **Store safety:** do not point smokes at the user's real `~/.abi/` path.

## Troubleshooting

| Symptom | Fix |
|---|---|
| build FAIL | `./tools/check.sh`; ensure nightly via `./tools/cargo.sh` |
| dyld FM shim missing | build from repo so `libabi_fm_shim.dylib` sits next to `abi` |
| MCP empty | JSON on stdout; logs on stderr — drop stderr when grepping tools |
