# ABI Framework Walkthrough

This walkthrough covers the Zig 0.17 modernization and expansion surfaces on the current branch.

## Toolchain

- Zig is pinned by `.zigversion` and `build.zig.zon`.
- On macOS, use `./build.sh ...`; it delegates to `tools/build.sh` and applies the repository's Darwin linker handling.
- Primary validation is `./build.sh check`.

## Build Commands

```bash
./build.sh check
./build.sh full-check
./build.sh cli
./build.sh mcp
zig build check-parity
zig build test-integration
zig build benchmarks
```

Feature flags default to enabled except `feat-mobile`:

```bash
./build.sh check -Dfeat-tui=false
./build.sh check -Dfeat-gpu=false
./build.sh check -Dfeat-ai=false
./build.sh check -Dfeat-wdbx=false
```

## CLI Walkthrough

Build the CLI first:

```bash
./build.sh cli
```

Inspect runtime backend status:

```bash
./zig-out/bin/abi backends
```

Run local model completion and record WDBX metadata:

```bash
./zig-out/bin/abi complete "Summarize the current ABI runtime status"
```

Run local AI routing:

```bash
./zig-out/bin/abi train "Summarize the current ABI runtime status"
```

Train local profiles with WDBX-backed storage:

```bash
./zig-out/bin/abi agent train all
```

List registered plugins:

```bash
./zig-out/bin/abi plugin list
```

Render diagnostics:

```bash
./zig-out/bin/abi dashboard
./zig-out/bin/abi tui
```

## GPU Backend

On macOS, the GPU module links Metal/Foundation/objc when `feat-gpu=true`. Runtime vector operations attempt to initialize a Metal context through Objective-C runtime messages and fall back to vectorized CPU operations if Metal setup fails.

`abi backends` reports:

- GPU backend name and whether the current runtime is accelerated.
- Native kernel status. A linked and initialized Metal context reports native kernels as available; otherwise the CPU fallback remains active.
- Shader, accelerator, and MLIR status.

## Router EMA Persistence

The AI profile router includes `AdaptiveModulator`, which maintains EMA-smoothed routing weights. The persistence key is:

```text
modulator:weights
```

When used with a WDBX `Store`, modulator weights can be loaded, updated, serialized as JSON, and stored back into the key-value surface.

## MCP Server

Build the MCP server:

```bash
./build.sh mcp
```

The MCP server uses JSON-RPC 2.0 over stdio and starts a loopback HTTP/SSE transport on `127.0.0.1:8080` when available. Request bodies are limited to 64KB. If port `8080` is already in use, set `ABI_MCP_HTTP_PORT` to another loopback port before launching `abi-mcp`.

Example stdio requests:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"ping"}
{"jsonrpc":"2.0","id":4,"method":"shutdown"}
```

Example tool call:

```json
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"ai_run","arguments":{"input":"hello"}}}
```

Example completion call:

```json
{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"ai_complete","arguments":{"input":"hello","model":"abi-local"}}}
```

HTTP endpoints:

- `GET /sse` returns an SSE endpoint event for `/message`.
- `POST /message` accepts a JSON-RPC request body and returns a JSON-RPC response.

Example alternate HTTP port:

```bash
ABI_MCP_HTTP_PORT=18080 ./zig-out/bin/abi-mcp
curl http://127.0.0.1:18080/sse
curl -X POST http://127.0.0.1:18080/message \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":7,"method":"ping"}'
```

## Verification Checklist

Run these before considering the branch complete:

```bash
./build.sh test --summary all
zig build check-parity
./build.sh check
./build.sh full-check
zig build test-integration
```

Run benchmarks explicitly when performance-sensitive GPU, WDBX, or routing code changes:

```bash
zig build benchmarks
```
