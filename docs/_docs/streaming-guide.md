---
title: Streaming Guide
description: AI streaming architecture with SSE, WebSocket, backpressure handling, and circuit breaker patterns
section: Guides
order: 93
permalink: /streaming-guide/
---

# Streaming Guide
## Summary
AI streaming architecture with SSE, WebSocket, backpressure handling, and circuit breaker patterns

## Architecture Overview

The AI streaming subsystem lives at `abi.ai.streaming` and provides a unified
abstraction for token-by-token LLM output delivery. It supports multiple transport
protocols and includes production-grade resilience patterns.

### Module Layout

```
src/features/ai/streaming/
  mod.zig              — Public API and re-exports
  stub.zig             — Disabled-feature stub
  generator.zig        — Core StreamingGenerator (pull-based)
  server.zig           — HTTP streaming server
  sse.zig              — Server-Sent Events encoder/decoder
  websocket.zig        — WebSocket frame handler
  backpressure.zig     — Flow control and rate adaptation
  circuit_breaker.zig  — Failure isolation per backend
  buffer.zig           — Ring buffer for token accumulation
  metrics.zig          — Streaming telemetry (latency, throughput)
  retry_config.zig     — Retry policies for local vs external backends
  recovery.zig         — Automatic recovery and reconnection
  session_cache.zig    — Session-level KV cache for multi-turn
  request_types.zig    — Shared request/response types
  backends/            — Per-backend streaming adapters
  formats/             — Output format adapters
```

## Server-Sent Events (SSE)

SSE is the default streaming transport. The `sse.zig` module provides both
encoding and decoding:

```zig
const sse = abi.ai.streaming.sse;

// Encode a stream event
var encoder = try sse.Encoder.init(allocator, .{});
defer encoder.deinit();

const frame = try encoder.encode(.{
    .event = "token",
    .data = "Hello",
    .id = "1",
});

// Decode an SSE stream
var decoder = try sse.Decoder.init(allocator, .{});
defer decoder.deinit();

const event = try decoder.decode(raw_bytes);
```

### SSE Wire Format

```
event: token
id: 42
data: {"text":"Hello","finish":false}

event: done
data: {"text":"","finish":true,"usage":{"prompt":12,"completion":87}}

```

## WebSocket Support

For bidirectional communication (e.g., interactive sessions), use the WebSocket
handler:

```zig
const websocket = abi.ai.streaming.websocket;

var handler = try websocket.WebSocketHandler.init(allocator, .{
    .max_frame_size = 64 * 1024,
    .ping_interval_ms = 30_000,
});
defer handler.deinit();

// Send a text frame
const frame = try handler.sendText("token data");
```

The WebSocket handler supports standard opcodes: text, binary, ping, pong, close.
Close codes follow RFC 6455 (normal=1000, going_away=1001, protocol_error=1002).

## Backpressure Handling

The backpressure module prevents fast producers from overwhelming slow consumers:

```zig
const backpressure = abi.ai.streaming.backpressure;

var controller = try backpressure.Controller.init(allocator, .{
    .high_water_mark = 1024,   // Pause production above this
    .low_water_mark = 256,     // Resume production below this
    .strategy = .drop_oldest,  // Or .block, .drop_newest
});
```

### Strategies

| Strategy | Behavior |
|---|---|
| `.block` | Producer blocks until consumer catches up |
| `.drop_oldest` | Discard oldest buffered tokens when full |
| `.drop_newest` | Discard incoming tokens when buffer is full |

## Circuit Breaker Pattern

Each streaming backend is wrapped in a circuit breaker that prevents cascading
failures:

```zig
const cb = abi.ai.streaming.circuit_breaker;

var breaker = cb.CircuitBreaker.init(.{
    .failure_threshold = 5,     // Open after 5 consecutive failures
    .timeout_ms = 60_000,       // Try half-open after 60s
    .success_threshold = 2,     // Close after 2 successes in half-open
});
```

### State Machine

```
CLOSED  -- failure_threshold reached -->  OPEN
OPEN    -- timeout_ms elapsed       -->  HALF_OPEN
HALF_OPEN -- success_threshold met  -->  CLOSED
HALF_OPEN -- any failure            -->  OPEN
```

## Retry Configuration

Different retry policies for local and external backends:

```zig
const retry = abi.ai.streaming.retry_config;

// Local backends (fast retry, short timeout)
const local = retry.StreamingRetryConfig.forLocalBackend();
// .max_retries = 3, .base_delay_ms = 100, .max_delay_ms = 1000

// External backends (slower retry, longer timeout)
const external = retry.StreamingRetryConfig.forExternalBackend();
// .max_retries = 5, .base_delay_ms = 500, .max_delay_ms = 30000
```

## Streaming Server

The streaming server exposes HTTP endpoints for SSE-based inference:

```bash
abi llm serve --port 8080 --model llama3
```

Clients connect via:
```
GET /v1/stream?prompt=hello
Accept: text/event-stream
```

The server manages concurrent sessions, applies backpressure per client, and uses
circuit breakers per backend.

## Generated Reference
## Overview

This guide is generated from repository metadata for **Guides** coverage and stays deterministic across runs.

## Build Snapshot

- Zig pin: `0.16.0-dev.2623+27eec9bd6`
- Main tests: `1261` pass / `5` skip / `1266` total
- Feature tests: `2082` pass / `2086` total

## Feature Coverage

- **llm** — Local LLM inference
  - Build flag: `enable_llm`
  - Source: `src/features/ai/facades/inference.zig`
  - Parent: `ai`

## Module Coverage

- `src/services/connectors/mod.zig` ([api](../api/connectors.html))

## Command Entry Points

- `abi agent` — Run AI agent (interactive or one-shot)
- `abi embed` — Generate embeddings from text (openai, mistral, cohere, ollama)
- `abi llm` — LLM inference (run, session, serve, providers, plugins, discover)
- `abi model` — Model management (list, download, remove, search)
- `abi ralph` — Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)
- `abi train` — Training pipeline (run, llm, vision, auto, self, resume, info)

## Validation Commands

- `zig build typecheck`
- `zig build check-docs`
- `zig build run -- gendocs --check`

## Navigation

- API Reference: [../api/](../api/)
- API App: [../api-app/](../api-app/)
- Plans Index: [../plans/index.md](../plans/index.md)
- Source Root: [GitHub src tree](https://github.com/donaldfilimon/abi/tree/master/src)

## Maintenance Notes
- This page is generated by `zig build gendocs`.
- Edit template source in `tools/gendocs/templates/docs/` for structural changes.
- Edit generator logic in `tools/gendocs/` for data model or rendering changes.


---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
