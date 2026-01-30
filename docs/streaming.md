---
title: "streaming"
tags: [ai, streaming, sse, websocket, inference]
---
# Streaming Inference API
> **Codebase Status:** Synced with repository as of 2026-01-26.

<p align="center">
  <img src="https://img.shields.io/badge/Module-Streaming-blue?style=for-the-badge&logo=lightning&logoColor=white" alt="Streaming Module"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge" alt="Production Ready"/>
  <img src="https://img.shields.io/badge/OpenAI-Compatible-green?style=for-the-badge" alt="OpenAI Compatible"/>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#endpoints">Endpoints</a> •
  <a href="#sse-streaming">SSE</a> •
  <a href="#websocket-streaming">WebSocket</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#cli-usage">CLI</a>
</p>

---

> **Developer Guide**: See [AI Guide](ai.md) for the full AI module documentation.
> **Framework**: Initialize ABI framework before using streaming features - see [Framework Guide](framework.md).

The **Streaming** module (`abi.ai.streaming`) provides real-time token streaming for LLM inference with Server-Sent Events (SSE) and WebSocket support.

## Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **SSE Streaming** | Unidirectional real-time token delivery | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **WebSocket Streaming** | Bidirectional communication with cancellation | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **OpenAI Compatibility** | Drop-in replacement for OpenAI chat completions | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Multi-Backend** | Route to local GGUF, OpenAI, Ollama, Anthropic | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Bearer Auth** | Token-based authentication | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Model Hot-Reload** | Swap models without server restart | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Heartbeat Keep-Alive** | Prevent proxy timeouts on long requests | ![Ready](https://img.shields.io/badge/-Ready-success) |

## Architecture

```
src/ai/streaming/
├── mod.zig           # Module entry point
├── server.zig        # HTTP server with SSE/WebSocket endpoints
├── sse.zig           # Server-Sent Events encoder
├── websocket.zig     # WebSocket frame encoder/decoder (RFC 6455)
├── generator.zig     # Token generation pipeline
├── buffer.zig        # Stream buffering utilities
├── backpressure.zig  # Flow control for slow clients
├── backends/         # Backend integrations
│   ├── mod.zig       # Backend router
│   ├── local.zig     # Local GGUF inference
│   ├── openai.zig    # OpenAI API
│   ├── ollama.zig    # Ollama API
│   └── anthropic.zig # Anthropic/Claude API
└── formats/          # Response formatters
    ├── mod.zig       # Format router
    ├── openai.zig    # OpenAI chat completion format
    └── abi.zig       # Custom ABI format
```

## Quick Start

### Starting the Server

```bash
# Start with a local GGUF model
abi llm serve -m ./models/llama-7b.gguf

# With authentication
abi llm serve -m ./models/llama-7b.gguf --auth-token my-secret-token

# Custom address and port
abi llm serve -m ./models/llama-7b.gguf -a 0.0.0.0:8080

# Pre-load model for faster first request
abi llm serve -m ./models/llama-7b.gguf --preload
```

### Making Requests

**SSE (curl):**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-token" \
  -d '{
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="my-secret-token"
)

stream = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**JavaScript (fetch):**
```javascript
const response = await fetch('http://localhost:8080/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer my-secret-token'
  },
  body: JSON.stringify({
    model: 'llama-7b',
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const text = decoder.decode(value);
  for (const line of text.split('\n')) {
    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
      const json = JSON.parse(line.slice(6));
      process.stdout.write(json.choices[0]?.delta?.content || '');
    }
  }
}
```

## Endpoints

### POST `/v1/chat/completions` (OpenAI-Compatible)

Drop-in replacement for OpenAI's chat completions API.

**Request:**
```json
{
  "model": "llama-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

**Response (SSE stream):**
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama-7b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama-7b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706000000,"model":"llama-7b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### POST `/api/stream` (ABI Custom Format)

Custom ABI endpoint with richer event types.

**Request:**
```json
{
  "prompt": "Explain machine learning",
  "backend": "local",
  "max_tokens": 500,
  "temperature": 0.8
}
```

**Response (SSE stream):**
```
data: {"type":"start","request_id":"req-123","model":"llama-7b","timestamp":1706000000}

data: {"type":"token","content":"Machine","index":0}

data: {"type":"token","content":" learning","index":1}

data: {"type":"token","content":" is","index":2}

data: {"type":"end","request_id":"req-123","total_tokens":150,"finish_reason":"stop"}
```

### GET `/api/stream/ws` (WebSocket)

Bidirectional WebSocket endpoint for advanced use cases.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080/api/stream/ws', [], {
  headers: { 'Authorization': 'Bearer my-secret-token' }
});

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'request',
    prompt: 'Write a poem about coding',
    max_tokens: 200
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case 'start':
      console.log('Stream started:', msg.request_id);
      break;
    case 'token':
      process.stdout.write(msg.content);
      break;
    case 'end':
      console.log('\nStream complete');
      break;
    case 'error':
      console.error('Error:', msg.message);
      break;
  }
};

// Cancel an in-progress stream
ws.send(JSON.stringify({ type: 'cancel' }));
```

**Message Types:**

| Type | Direction | Description |
|------|-----------|-------------|
| `request` | Client → Server | Start a new generation request |
| `cancel` | Client → Server | Cancel the current stream |
| `start` | Server → Client | Stream started acknowledgment |
| `token` | Server → Client | Generated token |
| `end` | Server → Client | Stream completed |
| `error` | Server → Client | Error occurred |

### GET `/health`

Health check endpoint (no authentication required by default).

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_streams": 3,
  "uptime_seconds": 3600
}
```

### POST `/admin/reload`

Hot-reload a new model without server restart.

**Request:**
```json
{
  "model_path": "/path/to/new-model.gguf"
}
```

**Behavior:**
- Waits for active streams to drain (30 second timeout)
- Unloads current model
- Loads new model
- Returns success/failure

**Response:**
```json
{
  "success": true,
  "previous_model": "llama-7b.gguf",
  "new_model": "mistral-7b.gguf",
  "load_time_ms": 2500
}
```

## SSE Streaming

Server-Sent Events provide unidirectional streaming from server to client.

### SSE Format

```
data: {"key": "value"}\n\n
```

- Each message is prefixed with `data: `
- Messages are separated by double newlines (`\n\n`)
- Stream ends with `data: [DONE]\n\n`

### Heartbeat Keep-Alive

Long-running streams send periodic heartbeats to prevent proxy timeouts:

```
: heartbeat

data: {"type":"token","content":"..."}

: heartbeat
```

Heartbeats are SSE comments (`:` prefix) that maintain the connection without affecting the data stream. Default interval is 15 seconds.

### Error Handling

Errors during streaming are sent as SSE events:

```
data: {"error":{"message":"Model inference failed","type":"backend_error","code":500}}

data: [DONE]
```

## WebSocket Streaming

WebSocket provides bidirectional communication with support for:

- Multiple requests per connection
- In-flight request cancellation
- Lower latency than SSE

### Frame Format

WebSocket uses RFC 6455 compliant framing with JSON payloads.

### Connection Lifecycle

1. **Connect** with `Authorization` header
2. **Send** request messages
3. **Receive** token stream
4. **Cancel** if needed
5. **Close** connection when done

### Concurrent Stream Limits

The server limits concurrent streams per connection (default: 100). Exceeding this returns an error:

```json
{"type":"error","code":"rate_limited","message":"Maximum concurrent streams exceeded"}
```

## Configuration

### Server Configuration

```zig
const config = abi.ai.streaming.ServerConfig{
    .address = "127.0.0.1:8080",
    .auth_token = "my-secret-token",
    .allow_health_without_auth = true,
    .default_backend = .local,
    .heartbeat_interval_ms = 15000,
    .max_concurrent_streams = 100,
    .enable_openai_compat = true,
    .enable_websocket = true,
    .default_model_path = "./models/llama-7b.gguf",
    .preload_model = true,
};

var server = try abi.ai.streaming.StreamingServer.init(allocator, config);
defer server.deinit();
try server.serve();
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `address` | `[]const u8` | `"127.0.0.1:8080"` | Listen address |
| `auth_token` | `?[]const u8` | `null` | Bearer token (null = no auth) |
| `allow_health_without_auth` | `bool` | `true` | Allow `/health` without auth |
| `default_backend` | `BackendType` | `.local` | Default inference backend |
| `heartbeat_interval_ms` | `u64` | `15000` | Heartbeat interval (0 = disabled) |
| `max_concurrent_streams` | `u32` | `100` | Max concurrent streams |
| `enable_openai_compat` | `bool` | `true` | Enable OpenAI endpoints |
| `enable_websocket` | `bool` | `true` | Enable WebSocket support |
| `default_model_path` | `?[]const u8` | `null` | Path to default model |
| `preload_model` | `bool` | `false` | Pre-load model on start |

### Backend Selection

Specify the backend in your request:

```json
{
  "backend": "openai",
  "model": "gpt-4",
  "messages": [...]
}
```

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `local` | Local GGUF model | Model file path |
| `openai` | OpenAI API | `ABI_OPENAI_API_KEY` |
| `ollama` | Ollama local server | Ollama running |
| `anthropic` | Anthropic/Claude API | `ABI_ANTHROPIC_API_KEY` |

## CLI Usage

### Start Server

```bash
# Basic usage
abi llm serve -m ./model.gguf

# Full options
abi llm serve \
  -m ./model.gguf \
  -a 0.0.0.0:8080 \
  --auth-token secret \
  --preload \
  --heartbeat 30000
```

### Run Benchmarks

```bash
# Quick streaming benchmark
abi bench streaming --preset quick

# Standard benchmark
abi bench streaming --preset standard

# Comprehensive benchmark
abi bench streaming --preset comprehensive
```

**Metrics measured:**
- TTFT (Time To First Token)
- Inter-token latency (P50/P90/P99)
- Throughput (tokens/second)
- SSE encoding overhead
- WebSocket framing overhead

## Programmatic Usage

### Zig Client Example

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create streaming client
    var client = try abi.ai.streaming.Client.init(allocator, .{
        .base_url = "http://localhost:8080",
        .auth_token = "my-token",
    });
    defer client.deinit();

    // Stream chat completion
    var stream = try client.chatCompletion(.{
        .model = "llama-7b",
        .messages = &[_]abi.ai.streaming.Message{
            .{ .role = .user, .content = "Hello!" },
        },
    });
    defer stream.deinit();

    while (try stream.next()) |chunk| {
        if (chunk.delta.content) |content| {
            std.debug.print("{s}", .{content});
        }
    }
}
```

## Best Practices

### 1. Use Heartbeats for Long Requests

Enable heartbeats when generating long responses to prevent proxy/load balancer timeouts:

```bash
abi llm serve -m ./model.gguf --heartbeat 15000
```

### 2. Pre-load Models for Low Latency

Use `--preload` to load the model at server startup instead of on first request:

```bash
abi llm serve -m ./model.gguf --preload
```

### 3. Implement Client-Side Reconnection

For production clients, implement exponential backoff reconnection:

```javascript
async function streamWithRetry(prompt, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await streamCompletion(prompt);
    } catch (e) {
      if (i === maxRetries - 1) throw e;
      await sleep(Math.pow(2, i) * 1000);
    }
  }
}
```

### 4. Use WebSocket for Interactive Apps

For chat applications with back-and-forth conversation, WebSocket provides:
- Lower latency (no HTTP overhead per message)
- Cancellation support
- Multiple requests per connection

### 5. Monitor with Health Endpoint

Poll `/health` for monitoring and alerting:

```bash
curl http://localhost:8080/health | jq '.active_streams'
```

## Troubleshooting

### Connection Refused

**Symptom:** `Connection refused` when connecting to server

**Solutions:**
1. Verify server is running: `abi llm serve -m ./model.gguf`
2. Check address binding: Use `0.0.0.0` instead of `127.0.0.1` for external access
3. Check firewall rules

### 401 Unauthorized

**Symptom:** `401 Unauthorized` response

**Solutions:**
1. Verify `Authorization: Bearer <token>` header is set
2. Ensure token matches server's `--auth-token` value
3. `/health` endpoint doesn't require auth by default

### Stream Timeout

**Symptom:** Connection closes before completion

**Solutions:**
1. Enable heartbeats: `--heartbeat 15000`
2. Increase proxy timeouts (nginx: `proxy_read_timeout`)
3. Check `max_concurrent_streams` limit

### Model Load Failures

**Symptom:** `ModelReloadFailed` error

**Solutions:**
1. Verify model file exists and is readable
2. Check model format (must be GGUF)
3. Ensure sufficient memory for model

## See Also

- [AI Guide](ai.md) - Full AI module documentation
- [LLM Guide](api/ai-llm.md) - LLM inference API reference
- [Model Management](models.md) - Downloading and managing models
- [Benchmarking Guide](benchmarking.md) - Performance testing
