---
title: "Inference"
description: "LLM inference, embeddings, vision, streaming, and transformer architecture in the ABI Inference module."
section: "AI"
order: 3
---

# Inference

The Inference module provides all inference-time AI capabilities: local LLM
execution with GGUF models, embedding generation, vision (multimodal)
processing, streaming output via SSE and WebSocket, and the transformer engine.

- **Build flag:** `-Denable-llm=true` (default: enabled)
- **Namespace:** `abi.inference`
- **Source:** `src/features/ai_inference/`

## Overview

The Inference module is the execution layer of the AI subsystem. While
[AI Core](ai-core.html) manages agents and tools, and
[Reasoning](ai-reasoning.html) handles higher-order logic, Inference is
responsible for running models and producing outputs.

Key capabilities:

- **LLM Engine** -- Load and run GGUF-format language models locally
- **Embeddings** -- Generate vector embeddings for text
- **Vision** -- Process and analyze images (multimodal inference)
- **Streaming** -- Server-Sent Events (SSE) and WebSocket streaming of tokens
- **Transformer** -- Configurable transformer architecture
- **Personas** -- Persona-aware inference (personas depend on embeddings)

## Quick Start

```zig
const abi = @import("abi");

// Initialize inference context with an LLM model
var ctx = try abi.inference.Context.init(allocator, .{
    .llm = .{ .model_path = "./models/llama-7b.gguf" },
    .embeddings = .{ .dimension = 768 },
});
defer ctx.deinit();

// Access the LLM engine
const llm = try ctx.getLlm();

// Access embeddings
const emb = try ctx.getEmbeddings();
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Module context managing LLM, embeddings, and personas |
| `LlmEngine` | Main LLM inference engine |
| `LlmModel` | A loaded language model |
| `LlmConfig` | Inference configuration (temperature, top-k, top-p, etc.) |
| `GgufFile` | GGUF model file parser and loader |
| `BpeTokenizer` | Byte-pair encoding tokenizer |
| `TransformerConfig` | Transformer model configuration |
| `TransformerModel` | Transformer model instance |
| `StreamingGenerator` | Token-by-token text generator |
| `StreamToken` | Individual token in a stream |
| `StreamState` | Current state of a streaming generation |
| `GenerationConfig` | Configuration for text generation (max tokens, stop sequences) |
| `ServerConfig` | Configuration for streaming server endpoints |
| `StreamingServer` | HTTP server for SSE / WebSocket streaming |
| `StreamingServerError` | Errors from the streaming server |
| `BackendType` | Streaming backend type (SSE, WebSocket) |

### Key Functions

| Function | Description |
|----------|-------------|
| `isEnabled() bool` | Returns `true` if inference is compiled in |
| `Context.init(allocator, config) !*Context` | Initialize inference with LLM and/or embeddings |
| `Context.getLlm() !*llm.Context` | Get the LLM sub-context |
| `Context.getEmbeddings() !*embeddings.Context` | Get the embeddings sub-context |

## LLM Inference

The LLM engine loads GGUF-format models and runs inference locally. GGUF is the
standard quantized model format used by llama.cpp and compatible tools.

```zig
const abi = @import("abi");

// Initialize with a GGUF model
var ctx = try abi.inference.Context.init(allocator, .{
    .llm = .{
        .model_path = "./models/llama-7b-q4_0.gguf",
    },
});
defer ctx.deinit();

const llm_ctx = try ctx.getLlm();
// Use llm_ctx for chat completions, text generation, etc.
```

### Supported Model Formats

The inference engine works with GGUF (GPT-Generated Unified Format) files,
which support various quantization levels for different memory/quality
trade-offs.

## Embeddings

Generate vector embeddings for semantic search, RAG pipelines, or similarity
matching:

```zig
var ctx = try abi.inference.Context.init(allocator, .{
    .embeddings = .{ .dimension = 768 },
});
defer ctx.deinit();

const emb_ctx = try ctx.getEmbeddings();
// Generate embeddings for text inputs
```

## Vision (Multimodal)

When built with `-Denable-vision=true`, the inference module can process images
alongside text for multimodal inference:

```zig
// Vision is conditionally compiled
const vision = abi.inference.vision;
// Process images through the vision pipeline
```

## Streaming

The streaming subsystem supports real-time token delivery over SSE and
WebSocket connections:

```zig
const streaming = abi.inference.streaming;

// Configure a streaming generator
const gen_config = streaming.GenerationConfig{
    .max_tokens = 512,
    // ...
};

// Or run a streaming server
const server_config = streaming.ServerConfig{
    // ...
};
```

### Streaming Backends

| Backend | Protocol | Use Case |
|---------|----------|----------|
| SSE | HTTP Server-Sent Events | Browser clients, curl |
| WebSocket | WebSocket frames | Bidirectional real-time apps |

## Transformer Architecture

The transformer module provides a configurable transformer implementation:

```zig
const transformer = abi.inference.transformer;

const config = transformer.TransformerConfig{
    // Configure layers, heads, dimensions, etc.
};
```

## Configuration

The inference module is configured through the `AiConfig` struct:

```zig
const config: abi.config.AiConfig = .{
    .llm = .{
        .model_path = "./models/llama-7b.gguf",
    },
    .embeddings = .{
        .dimension = 768,
    },
    .personas = .{
        // Persona settings (personas depend on embeddings)
    },
};

var ctx = try abi.inference.Context.init(allocator, config);
```

## CLI Commands

```bash
# LLM inference commands
zig build run -- llm info           # Show model info
zig build run -- llm generate       # Generate text
zig build run -- llm chat           # Interactive chat
zig build run -- llm bench          # Benchmark inference
zig build run -- llm download       # Download a model

# Embedding generation
zig build run -- embed --provider ollama "Your text here"
zig build run -- embed --provider openai "Your text here"

# Serve a model (alias for llm serve)
zig build run -- serve -m model.gguf
```

## Disabling at Build Time

```bash
# Compile without LLM inference
zig build -Denable-llm=false

# Compile without vision (keeps LLM and embeddings)
zig build -Denable-vision=false
```

When disabled, `Context.init()` returns `error.LlmDisabled` and `isEnabled()`
returns `false`. All type signatures are preserved by the stub module so
downstream code compiles cleanly.

## Examples

### Chat Completion

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize inference
    var ctx = try abi.inference.Context.init(allocator, .{
        .llm = .{ .model_path = "./models/llama-7b.gguf" },
    });
    defer ctx.deinit();

    const llm_ctx = try ctx.getLlm();
    // Perform chat completion through the LLM context
    _ = llm_ctx;
}
```

### Streaming Token Generation

```zig
const streaming = abi.inference.streaming;

// Create a streaming generator for real-time output
var generator = streaming.StreamingGenerator.init(allocator, .{
    .max_tokens = 256,
});
defer generator.deinit();
```

## Related

- [AI Overview](ai-overview.html) -- Architecture of all five AI modules
- [AI Core](ai-core.html) -- Agents, tools, prompts, personas
- [Training](ai-training.html) -- Model training pipelines
- [Reasoning](ai-reasoning.html) -- Abbey engine, RAG, orchestration
