---
name: run-abi
description: Build, launch, and drive the abi stable-Rust project — the `abi` CLI and the `abi-mcp` JSON-RPC server. Use when asked to run, start, build, smoke-test, or screenshot abi, drive its CLI subcommands, exercise the MCP server over stdio, or confirm a change works in the real binaries (not just the test suite).
---

# Run abi

`abi` is a **stable Rust 1.99.1** framework that builds two binaries: a CLI
(`target/debug/abi`) and an MCP server (`target/release/abi-mcp` in this smoke, JSON-RPC 2.0 over
stdio plus an automatically attempted custom loopback HTTP listener). The MCP
listener serves persistent MCP 2024-11-05 HTTP+SSE sessions plus a one-shot
direct-reply `POST /message` mode; it is not Streamable HTTP. The driver
builds and exercises the real binaries end-to-end.

The driver overrides inherited store settings with `ABI_WDBX_PATH=:memory:`
and `ABI_WDBX_PERSIST=0`. Its explicit-path WDBX round trip uses a scratch
store under `target/debug/`. A failed build stops before binary execution.
MCP runs through `mcp/launcher.sh stdio`; the driver rebuilds release MCP
because the launcher prefers that profile.
Builds explicitly select this checkout's `target/` directory, overriding an
inherited `CARGO_TARGET_DIR` so they refresh the binaries the driver executes.

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

- Stable Rust 1.99.1 via `rust-toolchain.toml`. **Always** use `./tools/cargo.sh`
  (never bare Homebrew `cargo`).
- Primary gate: `./tools/check.sh`.

## Build

```bash
./tools/cargo.sh build --locked -p abi-cli   # -> target/debug/abi
./tools/cargo.sh build --locked --release -p abi-mcp   # launcher-preferred profile
```

On arm64 macOS, building with the default `foundationmodels` feature also
produces `target/debug/libabi_fm_shim.dylib` next to the binaries.

## Drive the CLI directly

```bash
./target/debug/abi help
./target/debug/abi backends
./target/debug/abi scheduler status
ABI_WDBX_PATH=:memory: ./target/debug/abi complete "summarize scheduler status"
./target/debug/abi plugin list
```

## Drive the MCP server directly

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | ABI_WDBX_PATH=:memory: ABI_WDBX_PERSIST=0 ./mcp/launcher.sh stdio
```

Exactly **12** frozen tools. Always use `mcp/launcher.sh` so binary selection
and the FM dylib resolve together.

## Gotchas

- **GPU honesty:** `accelerated=false` means native kernels are not linked;
  CPU SIMD is the real path. Do not treat Metal-as-preferred as native kernels.
- **`complete` (no `--live`)** is fully local. `--live` is Anthropic-only for HTTP;
  `apple-fm --confirm` uses the FoundationModels shim when ready.
- **Store safety:** do not point smokes at the user's real `~/.abi/` path.

## Troubleshooting

| Symptom | Fix |
|---|---|
| build FAIL | `./tools/check.sh`; ensure the repository toolchain via `./tools/cargo.sh` |
| dyld FM shim missing | build from repo so `libabi_fm_shim.dylib` sits next to `abi` |
| MCP empty | JSON on stdout; logs on stderr — drop stderr when grepping tools |
