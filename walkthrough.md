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

Feature flags default to enabled except `feat-mobile` and `feat-metrics`; `feat-telemetry` is enabled by default. The check gate smoke-tests every feature-off stub and the real mobile module through `tools/check_feature_stubs.sh` using focused feature contracts plus feature-aware public contracts:

```bash
./build.sh check -Dfeat-tui=false
./build.sh check -Dfeat-gpu=false
./build.sh check -Dfeat-ai=false
./build.sh check -Dfeat-wdbx=false
zig build test-feature-contracts -Dfeat-mobile=true
zig build test-contracts -Dfeat-ai=false
zig build test-contracts -Dfeat-wdbx=false
zig build test-contracts -Dfeat-mobile=true
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

Run local model completion. The CLI opts into `CompletionRequest.store_result=true` and records WDBX metadata in a command-scoped in-memory store when WDBX is enabled; output includes `persisted=`, WDBX counts, vector IDs, `metadata_key`, and `block_id` when available:

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

List registered plugins, including plugin count plus generated version, target feature, entry point, and description metadata for the bundled example fixtures:

```bash
./zig-out/bin/abi plugin list
```

Render diagnostics:

```bash
./zig-out/bin/abi dashboard
./zig-out/bin/abi tui
```

Operate the WDBX runtime control surface:

```bash
./zig-out/bin/abi wdbx db init /tmp/abi-demo.wdbx.jsonl
./zig-out/bin/abi wdbx block insert /tmp/abi-demo.wdbx.jsonl Abbey "demo metadata"
./zig-out/bin/abi wdbx db verify /tmp/abi-demo.wdbx.jsonl
./zig-out/bin/abi wdbx cluster status
./zig-out/bin/abi wdbx compute info
./zig-out/bin/abi wdbx secure demo
./zig-out/bin/abi wdbx api serve 8081
```

`abi wdbx` is a local runtime/demo namespace. Snapshot, WAL, block, query, benchmark, GPU, and loopback REST paths operate local state; `cluster`, `compute`, and `secure` demonstrate in-process consensus, backend selection with CPU fallback, int8 quantization, and additive aggregation. They do not prove distributed clustering, native NPU/TPU execution, learned compression, or full homomorphic encryption.

## GPU Backend

On macOS, the GPU module links Metal/Foundation/objc when `feat-gpu=true`. Runtime vector operations attempt to initialize a Metal context through Objective-C runtime messages and fall back to vectorized CPU operations if Metal setup fails. With `-Dfeat-gpu=false`, the public GPU API remains available through `src/features/gpu/stub.zig` and reports a deterministic CPU fallback.

`abi backends` reports:

- GPU backend name and whether the current runtime is accelerated.
- Native kernel status. A linked and initialized Metal context reports native kernels as available; otherwise the CPU fallback remains active.
- Shader, accelerator, and MLIR status.

## Router EMA Persistence

The AI profile router includes `AdaptiveModulator`, which maintains EMA-smoothed routing weights. The persistence key is:

```text
modulator:weights
```

When used with a WDBX `Store`, modulator weights can be loaded, updated, serialized as JSON, and stored back into the key-value surface. API callers must set `CompletionRequest.store_result=true` to persist completion vectors/metadata; `store_result=false`, invalid completion input, and disabled WDBX leave the store unchanged. WDBX manifests now report key-value, vector, block, spatial-record, vector-dimension, next-vector-id, backend, and execution-mode fields; disabled builds keep a compatible manifest shape with `"disabled":true`. JSONL snapshots include an integrity line, and the write-ahead log replays framed records with corruption detection.

## MCP Server

Build the MCP server:

```bash
./build.sh mcp
```

The MCP server uses JSON-RPC 2.0 over stdio and starts a loopback HTTP/SSE transport on `127.0.0.1:8080` when available. Request bodies are limited to 64KB. If port `8080` is already in use, set `ABI_MCP_HTTP_PORT` to another loopback port before launching `abi-mcp`; invalid overrides fall back to `8080`, and bind failure leaves stdio running.

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

Example completion call. Like the CLI, MCP `ai_complete` uses a request-scoped in-memory WDBX store and reports whether persistence occurred:

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

## Connector Validation

Connector local paths are deterministic; live paths require explicit `.live` transport calls.

- Discord validates printable non-whitespace tokens, numeric snowflake-like client/channel/author IDs, and the 2000-byte message limit.
- Twilio validates account SIDs as `AC` plus 32 hex characters, auth tokens as 32 hex characters, non-empty base URL, non-zero timeout, and explicit `.live` transport selection. Its ConversationRelay parser accepts Twilio-style aliases such as `event`, `callSid`, `from`, and camelCase memory/intelligence fields, rejects wrong-typed fields, and escapes TwiML/XML plus URL-encoded form payloads before local/live dispatch.
- OpenAI and Anthropic local streaming helpers return deterministic SSE-like responses and never call the network unless a live method is used.

## Verification Checklist

Run these before considering the branch complete:

```bash
zig build test --summary all
zig build check-parity
./build.sh check
./build.sh full-check   # check + integration tests + benchmarks + TUI smoke
zig build test-integration
```

Run benchmarks explicitly when performance-sensitive GPU, WDBX, or routing code changes:

```bash
zig build benchmarks
```
