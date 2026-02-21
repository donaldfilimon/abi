# AI Module

AI module providing LLM inference, agents, embeddings, and training capabilities.

## Features

- **LLM Inference**: Local (GGUF) and API-based language model inference
- **Agents**: Autonomous AI agents with tool use and memory
- **Embeddings**: Vector embedding generation for semantic search
- **Training**: Model fine-tuning, vision/multimodal (ViT, CLIP), and auto-train (Abbey, Aviva, Abi)
- **Streaming**: SSE/WebSocket inference and session recovery

## Sub-modules
//!
| Module | Description |
|--------|-------------|
| `mod.zig` | Public API entry point with Context struct |
| `llm/` | Local LLM inference (GGUF), tokenizers, sampling |
| `embeddings/` | Embedding generation |
| `agents/` | Agent runtime and tool use |
| `training/` | Training pipelines (LLM, vision, multimodal, auto-train) |
| `streaming/` | SSE/WebSocket streaming server |
| `orchestration/` | Multi-model routing, ensemble, fallback |
| `rag/` | Retrieval-augmented generation |
| `documents/` | Document parsing, segmentation, entities |
| `memory/` | Agent memory and persistence |
| `models/` | Model management and downloads |
| `explore/` | Codebase exploration |

## Architecture

This module is a primary feature module. The implementation lives in `src/features/ai/` with a feature-gated stub in `stub.zig` for builds with `-Denable-ai=false`.

## Usage

```zig
const abi = @import("abi");

var fw = try abi.init(allocator, .{
    .ai = .{
        .llm = .{ .model_path = "./models/llama.gguf" },
        .embeddings = .{},
    },
});
defer fw.deinit();

const ai_ctx = try fw.getAi();
const response = try ai_ctx.llm.generate("Hello, world!", .{});
defer allocator.free(response);
```

## LLM Connectors

| Connector | Description | Environment / config |
|-----------|-------------|----------------------|
| Local GGUF | Load local GGUF models | `model_path` |
| Ollama | Ollama server | `ABI_OLLAMA_HOST`, `ABI_OLLAMA_MODEL` |
| OpenAI | OpenAI API | `ABI_OPENAI_API_KEY` |
| Anthropic | Claude API | `ABI_ANTHROPIC_API_KEY` |
| LM Studio | Local OpenAI-compatible | `ABI_LM_STUDIO_HOST`, `ABI_LM_STUDIO_MODEL` |
| vLLM | Local OpenAI-compatible | `ABI_VLLM_HOST`, `ABI_VLLM_MODEL` |
| HuggingFace | HuggingFace API | `ABI_HF_API_TOKEN` |

## Build options

- **Enable**: `-Denable-ai=true` (default).
- **Sub-features**: `-Denable-llm`, `-Denable-vision`, `-Denable-explore` (all require `-Denable-ai`).

## See also

- [AI Documentation](../../../docs/_docs/ai-overview.md)
- [API Reference](../../../docs/api/)
- [CLAUDE.md](../../../CLAUDE.md) — Connectors and env vars


