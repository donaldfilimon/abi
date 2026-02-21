---
title: "LLM Connectors"
description: "9 LLM providers, Discord, and job scheduling"
section: "Services"
order: 1
---

# LLM Connectors

The connectors module provides unified access to 9 LLM providers, a Discord
bot client, and a local job scheduler. All connectors follow a consistent
pattern: environment-based configuration, secure API key handling, and
zero-allocation availability checks.

- **Namespace:** `abi.connectors`
- **Source:** `src/services/connectors/`

Connectors are always-available infrastructure (no feature flag needed). They
are compiled in whenever their parent module is present.

## Overview

### LLM Providers

| Provider | Namespace | API Style | Default Host |
|----------|-----------|-----------|-------------|
| OpenAI | `abi.connectors.openai` | Chat Completions | `https://api.openai.com/v1` |
| Anthropic | `abi.connectors.anthropic` | Messages API | `https://api.anthropic.com` |
| Ollama | `abi.connectors.ollama` | Chat/Generate | `http://127.0.0.1:11434` |
| HuggingFace | `abi.connectors.huggingface` | Inference API | `https://api-inference.huggingface.co` |
| Mistral | `abi.connectors.mistral` | OpenAI-compatible | `https://api.mistral.ai/v1` |
| Cohere | `abi.connectors.cohere` | Chat/Embed/Rerank | `https://api.cohere.ai/v1` |
| LM Studio | `abi.connectors.lm_studio` | OpenAI-compatible | `http://localhost:1234` |
| vLLM | `abi.connectors.vllm` | OpenAI-compatible | `http://localhost:8000` |
| MLX | `abi.connectors.mlx` | OpenAI-compatible | `http://localhost:8080` |

### Other Connectors

| Connector | Namespace | Description |
|-----------|-----------|-------------|
| Discord | `abi.connectors.discord` | Discord bot REST client |
| Local Scheduler | `abi.connectors.local_scheduler` | Job scheduling |

### Shared Types

All connectors share common types defined in `shared.zig`:

| Type | Description |
|------|-------------|
| `ChatMessage` | Message with `role` and `content` fields |
| `Role` | Constants: `SYSTEM`, `USER`, `ASSISTANT`, `FUNCTION`, `TOOL` |
| `ConnectorError` | Common errors: `MissingApiKey`, `ApiRequestFailed`, `InvalidResponse`, `RateLimitExceeded`, `Timeout`, `OutOfMemory` |

## Quick Start

```zig
const abi = @import("abi");
const connectors = abi.connectors;

// Load config from environment variables
if (try connectors.tryLoadOpenAI(allocator)) |config| {
    defer config.deinit(allocator);
    // Use config.api_key, config.base_url, config.model
}

// Check availability without allocation
if (connectors.shared.envIsSet("ABI_OPENAI_API_KEY")) {
    // OpenAI is configured
}
```

## Environment Variables

| Variable | Provider | Required |
|----------|----------|----------|
| `ABI_OPENAI_API_KEY` or `OPENAI_API_KEY` | OpenAI | Yes |
| `ABI_OPENAI_BASE_URL` or `OPENAI_BASE_URL` | OpenAI | No |
| `ABI_OPENAI_MODEL` or `OPENAI_MODEL` | OpenAI | No (default: `gpt-4`) |
| `ABI_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY` | Anthropic | Yes |
| `ABI_OLLAMA_HOST` or `OLLAMA_HOST` | Ollama | No (default: `http://127.0.0.1:11434`) |
| `ABI_OLLAMA_MODEL` or `OLLAMA_MODEL` | Ollama | No |
| `ABI_HF_API_TOKEN` or `HF_API_TOKEN` | HuggingFace | Yes |
| `ABI_LM_STUDIO_HOST` | LM Studio | No (default: `http://localhost:1234`) |
| `ABI_LM_STUDIO_MODEL` | LM Studio | No |
| `ABI_LM_STUDIO_API_KEY` | LM Studio | No |
| `ABI_VLLM_HOST` | vLLM | No (default: `http://localhost:8000`) |
| `ABI_VLLM_MODEL` | vLLM | No |
| `ABI_VLLM_API_KEY` | vLLM | No |
| `ABI_MLX_HOST` | MLX | No (default: `http://localhost:8080`) |
| `ABI_MLX_MODEL` | MLX | No |
| `ABI_MLX_API_KEY` | MLX | No |
| `DISCORD_BOT_TOKEN` | Discord | Yes (for Discord) |

## Provider Examples

### OpenAI

```zig
const openai = abi.connectors.openai;

// Load from environment
var config = try openai.loadFromEnv(allocator);
defer config.deinit(allocator);

// Or use tryLoad (returns null if API key not set)
if (try abi.connectors.tryLoadOpenAI(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // cfg.api_key, cfg.base_url, cfg.model ("gpt-4")
}
```

### Anthropic

```zig
const anthropic = abi.connectors.anthropic;

var config = try anthropic.loadFromEnv(allocator);
defer config.deinit(allocator);
// config.model defaults to "claude-3-5-sonnet-20241022"
// config.max_tokens defaults to 4096
```

### Ollama (Local)

```zig
const ollama = abi.connectors.ollama;

// Ollama does not require an API key
var config = try ollama.loadFromEnv(allocator);
defer config.deinit(allocator);
// config.host defaults to "http://127.0.0.1:11434"
```

### LM Studio / vLLM / MLX (Local OpenAI-compatible)

LM Studio, vLLM, and MLX all use OpenAI-compatible `/v1/chat/completions`
endpoints, making them interchangeable for chat workloads:

```zig
// LM Studio
if (try abi.connectors.tryLoadLMStudio(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // Uses http://localhost:1234/v1/chat/completions
}

// vLLM (high-throughput serving)
if (try abi.connectors.tryLoadVLLM(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // Uses http://localhost:8000/v1/chat/completions
}

// MLX (Apple Silicon optimized)
if (try abi.connectors.tryLoadMLX(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // Uses http://localhost:8080/v1/chat/completions
}
```

### HuggingFace

```zig
if (try abi.connectors.tryLoadHuggingFace(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // Hosted inference API
}
```

### Mistral

```zig
if (try abi.connectors.tryLoadMistral(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // OpenAI-compatible API at api.mistral.ai
}
```

### Cohere

```zig
if (try abi.connectors.tryLoadCohere(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // Chat, embeddings, and reranking
}
```

### Discord

```zig
if (try abi.connectors.tryLoadDiscord(allocator)) |cfg| {
    defer cfg.deinit(allocator);
    // Discord bot REST client
}
```

## Availability Checks

Every connector provides zero-allocation availability checks that read
environment variables without heap allocation:

```zig
const shared = abi.connectors.shared;

// Check a single variable
if (shared.envIsSet("ABI_OPENAI_API_KEY")) {
    // OpenAI is configured
}

// Check multiple variables (returns true if any is set)
if (shared.anyEnvIsSet(&.{ "ABI_OPENAI_API_KEY", "OPENAI_API_KEY" })) {
    // OpenAI key available under either name
}
```

These use a stack-allocated buffer for the null-terminated string and call
libc `getenv()` directly -- no heap allocation is needed.

## Security

All connectors securely wipe API keys from memory before freeing:

```zig
// Config.deinit() calls shared.secureFree() on the API key,
// which uses std.crypto.secureZero before freeing the allocation.
var config = try openai.loadFromEnv(allocator);
defer config.deinit(allocator);  // API key is securely wiped here
```

The `model_owned: bool` field on Config structs tracks whether the model string
was heap-allocated (from environment) or is a comptime default, preventing
use-after-free on `deinit`.

## Retry and Error Handling

Shared retry utilities provide exponential backoff with jitter:

```zig
const shared = abi.connectors.shared;

// Calculate delay: base_ms * 2^attempt, capped at max_ms
const delay = shared.exponentialBackoff(attempt, 1000, 60000);

// With jitter (+-25%) to prevent thundering herd
const jittered = shared.calculateRetryDelay(attempt, 1000, 60000);

// Check if a status code is retryable (429 or 5xx)
if (shared.isRetryableStatus(status)) {
    // Retry the request
}
```

## Helper Functions

The connectors module provides JSON encoding helpers:

```zig
// Encode chat messages as JSON
var buf = std.ArrayListUnmanaged(u8).empty;
defer buf.deinit(allocator);
try shared.encodeMessageArray(allocator, &buf, &messages);

// Encode string arrays as JSON
try shared.encodeStringArray(allocator, &buf, &strings);

// HTTP status helpers
shared.isSuccessStatus(200);     // true
shared.isRateLimitStatus(429);   // true
shared.isClientError(404);       // true
shared.isServerError(503);       // true
```

## Related

- [AI Core](ai-core.html) -- Agent framework that uses connectors for LLM access
- [MCP Server](mcp.html) -- MCP tools that can leverage connectors
- [Auth & Security](auth.html) -- API key management and rate limiting
- [Deployment](deployment.html) -- Environment variable configuration for production

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
