---
title: Connectors Guide
description: Connector configuration, environment variables, auto-discovery, and custom plugin connectors
section: Guides
order: 94
permalink: /connectors-guide/
---

# Connectors Guide
## Summary
Connector configuration, environment variables, auto-discovery, and custom plugin connectors

## Overview

Connectors are adapter modules that translate ABI's internal request format into
provider-specific wire protocols. Each connector lives in
`src/services/connectors/<name>.zig` and implements a standard interface: config
loading from environment variables, client initialization, request dispatch, and
secure teardown.

All connectors securely wipe API keys from memory using `std.crypto.secureZero`
before freeing to prevent memory forensics attacks.

## Connector Matrix

| Connector | Type | Default Endpoint | Auth Env Var | Protocol |
|---|---|---|---|---|
| **Anthropic** | Remote API | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | Messages API |
| **OpenAI** | Remote API | `https://api.openai.com` | `OPENAI_API_KEY` | Chat Completions |
| **Ollama** | Local | `http://127.0.0.1:11434` | — | Ollama JSON |
| **llama.cpp** | Local | `http://127.0.0.1:8080` | — | llama.cpp HTTP |
| **MLX** | Local | `http://127.0.0.1:5001` | — | mlx-lm server |
| **LM Studio** | Local | `http://127.0.0.1:1234` | — | OpenAI-compatible |
| **vLLM** | Local | `http://127.0.0.1:8000` | — | OpenAI-compatible |
| **HuggingFace** | Remote API | `https://api-inference.huggingface.co` | `HUGGINGFACE_API_KEY` | Inference API |
| **Mistral** | Remote API | `https://api.mistral.ai` | `MISTRAL_API_KEY` | OpenAI-compatible |
| **Cohere** | Remote API | `https://api.cohere.ai` | `COHERE_API_KEY` | Chat/Embed/Rerank |
| **Discord** | Bot | `https://discord.com/api` | `DISCORD_BOT_TOKEN` | Discord Gateway |

## Environment Variable Configuration

Each remote connector reads credentials from environment variables at load time.
Local connectors auto-detect running servers by probing their default endpoints.

### Required Variables (Remote Providers)

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# HuggingFace
export HUGGINGFACE_API_KEY="hf_..."

# Mistral
export MISTRAL_API_KEY="..."

# Cohere
export COHERE_API_KEY="..."

# Discord
export DISCORD_BOT_TOKEN="..."
```

### Optional Overrides

```bash
# Override default endpoint URLs
export OPENAI_BASE_URL="https://my-proxy.example.com"
export ANTHROPIC_BASE_URL="https://my-proxy.example.com"
export OLLAMA_HOST="http://192.168.1.100:11434"
export LM_STUDIO_HOST="http://192.168.1.100:1234"
export VLLM_HOST="http://192.168.1.100:8000"
export MLX_HOST="http://192.168.1.100:5001"
export LLAMA_CPP_HOST="http://192.168.1.100:8080"
```

## Connector Architecture

### Loading Pattern

Every connector follows the same two-phase pattern:

```zig
const connectors = @import("abi").connectors;

// Phase 1: Load config from environment
if (try connectors.tryLoadOpenAI(allocator)) |config| {
    // Phase 2: Create client from config
    var client = try connectors.openai.Client.init(allocator, config);
    defer client.deinit();

    // Use client
    const response = try client.chatCompletion(allocator, .{
        .model = "gpt-4",
        .messages = &.{.{ .role = .user, .content = "Hello" }},
    });
}
```

### Shared Infrastructure

All connectors share common utilities from `src/services/connectors/shared.zig`:

- **HTTP client** — connection pooling, timeout management, retry logic
- **Auth header injection** — `Authorization: Bearer` for API keys
- **Response parsing** — JSON deserialization with error extraction
- **Secure cleanup** — zero-fill sensitive data on deallocation

### Local Connector Auto-Discovery

Local connectors (Ollama, llama.cpp, MLX, LM Studio, vLLM) support automatic
discovery. Run `abi llm discover` to probe all default endpoints:

```bash
$ abi llm discover
Scanning local providers...
  ollama      http://127.0.0.1:11434  [OK] 3 models
  lm_studio   http://127.0.0.1:1234   [OK] 1 model
  mlx         http://127.0.0.1:5001   [OFFLINE]
  llama_cpp   http://127.0.0.1:8080   [OFFLINE]
  vllm        http://127.0.0.1:8000   [OFFLINE]
```

## Adding a Custom Connector

For custom LLM backends, use the plugin system:

```bash
# HTTP plugin — any OpenAI-compatible server
abi llm plugins add --name my-backend --url http://localhost:9090

# Native plugin — Zig shared library
abi llm plugins add --name my-native --path ./libmyprovider.dylib
```

Plugin connectors appear in the provider router alongside built-in connectors.

## Generated Reference
## Overview

This guide is generated from repository metadata for **Guides** coverage and stays deterministic across runs.

## Build Snapshot

- Zig pin: `0.16.0-dev.2637+6a9510c0e`
- Main tests: `1290` pass / `6` skip / `1296` total
- Feature tests: `2360` pass / `5` skip / `2365` total

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
