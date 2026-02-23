---
title: LLM Inference Guide
description: Provider router architecture, backend selection, fallback chains, and CLI usage for LLM inference
section: Guides
order: 90
permalink: /llm-inference-guide/
---

# LLM Inference Guide
## Summary
Provider router architecture, backend selection, fallback chains, and CLI usage for LLM inference

## Provider Router Architecture

The LLM subsystem uses a **provider router** that selects and dispatches inference
requests across multiple backends. At startup the router discovers available providers
and builds a priority-ordered routing table. If a primary provider fails, the router
walks a configurable **fallback chain** until a healthy backend responds.

### Provider Discovery

Run `abi llm discover` to scan the local machine and reachable network for active
backends. The command probes each endpoint's health URL and model list, then writes
a cached provider manifest to the platform-specific ABI app root (for example
`~/Library/Application Support/abi/providers.json` on macOS,
`$XDG_CONFIG_HOME/abi/providers.json` or `~/.config/abi/providers.json` on Linux, and
`%APPDATA%\\abi\\providers.json` on Windows), with lazy fallback reads from legacy
`~/.abi/providers.json` (or `%USERPROFILE%\\.abi\\providers.json`).

### Routing Order

The default routing priority (highest first):

| Priority | Backend | Type | Endpoint |
|---|---|---|---|
| 1 | `local_gguf` | Local | Direct weight file load |
| 2 | `llama_cpp` | Local | `http://127.0.0.1:8080` |
| 3 | `mlx` | Local | `http://127.0.0.1:5001` |
| 4 | `ollama` | Local | `http://127.0.0.1:11434` |
| 5 | `lm_studio` | Local | `http://127.0.0.1:1234` |
| 6 | `vllm` | Local | `http://127.0.0.1:8000` |
| 7 | `anthropic` | Remote | `https://api.anthropic.com` |
| 8 | `openai` | Remote | `https://api.openai.com` |
| 9 | `plugin_http` | Plugin | User-defined URL |
| 10 | `plugin_native` | Plugin | Zig `.so`/`.dylib` |

Override order with `--backend <name>` or `--fallback <name1>,<name2>,...`.

## CLI Commands

### `abi llm run`

One-shot generation through the provider router:

```bash
# Local GGUF file
abi llm run --model ./model.gguf --prompt "Explain quick sort"

# Named model with explicit fallback chain
abi llm run --model llama3 --prompt "Hello" --fallback mlx,ollama

# Remote provider
abi llm run --model claude-3 --backend anthropic --prompt "Draft a plan"
```

### `abi llm session`

Interactive multi-turn session with conversation history:

```bash
abi llm session --model llama3 --backend ollama
abi llm session --model gpt-4 --backend openai
```

### `abi llm discover`

Auto-discover available LLM providers and display their status:

```bash
abi llm discover
```

### `abi llm providers`

Show provider availability and routing order:

```bash
abi llm providers
```

### `abi llm serve`

Start a streaming HTTP server that exposes LLM inference via SSE:

```bash
abi llm serve --port 8080 --model llama3
```

### `abi llm plugins`

Manage HTTP and native provider plugins:

```bash
abi llm plugins list
abi llm plugins add --name my-backend --url http://localhost:9090
```

## Backend Selection

The router evaluates backends in this order:

1. **Explicit** — `--backend <name>` bypasses discovery.
2. **Fallback chain** — `--fallback mlx,ollama,openai` tries each in sequence.
3. **Auto** — (default) uses discovery results and priority table above.

If all candidates fail, the router returns `error.NoAvailableProvider`.

## Structured Messages

The LLM subsystem uses a unified message format across all providers:

```zig
const Message = struct {
    role: enum { system, user, assistant, tool },
    content: []const u8,
    name: ?[]const u8 = null,
};
```

Providers that use different wire formats (Anthropic Messages API, OpenAI Chat
Completions, Ollama JSON) are translated at the connector layer.

## Generated Reference
## Overview

This guide is generated from repository metadata for **Guides** coverage and stays deterministic across runs.

## Build Snapshot

- Zig pin: `0.16.0-dev.2623+27eec9bd6`
- Main tests: `1290` pass / `6` skip / `1296` total
- Feature tests: `2360` pass / `2365` total

## Feature Coverage

- **embeddings** — Vector embeddings generation
  - Build flag: `enable_ai`
  - Source: `src/features/ai/mod.zig`
  - Parent: `ai`
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
