---
name: run-abi
description: Build, launch, and drive the abi Zig project — the `abi` CLI and the `abi-mcp` JSON-RPC server. Use when asked to run, start, build, smoke-test, or screenshot abi, drive its CLI subcommands, exercise the MCP server over stdio, or confirm a change works in the real binaries (not just the test suite).
---

# Run abi

`abi` is a modular Zig framework that builds two binaries: a CLI (`zig-out/bin/abi`)
and an MCP server (`zig-out/bin/abi-mcp`, JSON-RPC 2.0 over stdio + optional
loopback HTTP/SSE). Both are **non-interactive and headless-friendly** — the
driver builds them and drives the real binaries end-to-end. The diagnostics TUI
also has a clean non-TTY one-shot mode; use tmux only when you need to exercise
the interactive refresh loop (see Gotchas).

**Paths below are relative to the repo root** (`<unit>/`). The driver lives at
`.agents/skills/run-abi/smoke.sh` and resolves the repo root from its own location,
so you can run it from any cwd.

## Run (agent path) — the driver

One command builds both binaries and drives them with real input — CLI
subcommands (help, backends, scheduler, complete, plugin list), a WDBX store round-trip
(init → insert → query), and the MCP server over stdio with real JSON-RPC
(initialize + tools/list + tools/call). It checks exit codes, greps output for
expected markers, prints `pass=N fail=N`, and writes a full transcript.

```bash
./.agents/skills/run-abi/smoke.sh
```

Expected tail on success (exit 0):

```
=== summary: pass=16 fail=0 ===
transcript: <repo>/zig-out/run-abi-smoke.txt
SMOKE OK
```

The transcript lands at `zig-out/run-abi-smoke.txt` and a scratch WDBX store at
`zig-out/smoke-memory.jsonl`. Read the transcript to see every command, its
output, and exit code. A non-zero exit equals the number of failed checks.

## Prerequisites

- macOS (this repo is Darwin-first; `build.zig` links Metal when `feat-gpu=true`).
- The pinned dev Zig must already be on `PATH`. `.zigversion` pins
  `0.17.0-dev.1252+e4b325c19`; the tree also compiles on nearby Zig master nightlies
  (verify with `zig version` / the `zig-newest-skills` driver). `./build.sh` does **not**
  switch or enforce the pin — it runs whatever `zig` is on `PATH` and just echoes
  `Using Zig: …`. Select the toolchain with zvm/zigup first. Zig 0.16 will not
  compile this tree.

```bash
zig version    # must be a 0.17.0-dev build
```

## Build

The driver builds for you, but to build the binaries directly:

```bash
./build.sh cli    # -> zig-out/bin/abi
./build.sh mcp    # -> zig-out/bin/abi-mcp
```

Cold builds are slow (full feature graph + Metal link). Builds are incremental
against `.zig-cache/`; a warm rebuild + full smoke run finishes in ~1s.

## Drive the CLI directly

All of these were run as part of the smoke pass:

```bash
./zig-out/bin/abi help
./zig-out/bin/abi backends                 # GPU/accelerator/shader/MLIR report
./zig-out/bin/abi scheduler status
./zig-out/bin/abi complete "summarize scheduler status"   # -> model=claude-fable-5 ...
./zig-out/bin/abi plugin list              # CLI sees all 16 bundled plugins
./zig-out/bin/abi wdbx db init zig-out/smoke-memory.jsonl
./zig-out/bin/abi wdbx block insert zig-out/smoke-memory.jsonl abi '{"note":"hi"}'
./zig-out/bin/abi wdbx query zig-out/smoke-memory.jsonl
```

## Drive the MCP server directly

The server reads newline-delimited JSON-RPC on stdin and writes responses to
stdout. Pipe requests in; `stderr` carries logs (redirect with `2>/dev/null`):

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"plugin_list","arguments":{}}}' \
  | ./zig-out/bin/abi-mcp stdio 2>/dev/null
```

Returns the `serverInfo` handshake then the tool result. Other no-arg tools to
probe: `scheduler_info`, `gpu_status`, `wdbx_stats`. The smoke also calls
`plugin_list` and expects the same 16 bundled plugins as the CLI. The server exposes 12 tools;
`tools/list` enumerates them with their input schemas.

## Run (human path) — the TUI

The diagnostics dashboard is interactive on a real terminal:

```bash
./zig-out/bin/abi tui        # or: abi dashboard / abi --tui
```

Run it in your own terminal and quit with the in-app key. For headless checks,
use the dashboard one-shot smoke (`abi dashboard < /dev/null`) or the tmux-based
`run-tui` skill for the interactive path.

## Gotchas

- **`abi tui`/`dashboard`/`--tui` are interactive on a TTY.** With stdin piped or
  `/dev/null`, they should render once and exit cleanly through the non-TTY
  fallback. Use a pty (tmux `send-keys`/`capture-pane`) to drive the live loop.
- **CLI and MCP plugin lists should match.** Both surfaces load the 16 bundled
  plugin manifests; drift means the shared plugin manager/registry wiring changed.
- **`accelerated=false` is normal.** `backends` reports `metal: available=true
  accelerated=false` — Metal is linked at build time but native kernels aren't,
  so it runs the deterministic vectorized CPU fallback. Not a failure.
- **`./build.sh` does not honor `.zigversion`.** It runs whatever `zig` is on
  `PATH`. If a build fails with std API errors, check `zig version` first.
- **`complete` (no `--live`) is fully local** — it routes to the local model and
  records WDBX metadata; it does not call any remote provider.

## Troubleshooting

- `command not found: timeout` (when probing the TUI on macOS): `timeout` isn't a
  default macOS command. Use a background-process + `kill` pattern, or `gtimeout`
  from coreutils.
- Smoke prints `FATAL: binaries not produced`: a build step failed — re-run
  `./build.sh cli` / `./build.sh mcp` and read the compiler error; usually a
  wrong `zig` on `PATH`.
- MCP responses look empty: logs go to `stderr`. Keep `2>/dev/null` (or inspect
  stderr separately) so JSON on stdout is clean.
