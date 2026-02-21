---
title: "AI Overview"
description: "Architecture overview of the ABI framework's five independent AI modules: ai, ai_core, inference, training, and reasoning."
section: "AI"
order: 1
---

# AI Overview

The ABI framework provides a modular AI subsystem split into **five independent
modules**, each gated by its own build flag and compiled at comptime. This
architecture allows you to include only the AI capabilities your application
needs while paying zero binary overhead for disabled features.

## Module Architecture

All five modules live under `src/features/` and follow the standard ABI pattern
of `mod.zig` (real implementation) + `stub.zig` (returns `error.FeatureDisabled`
when the module is compiled out).

| Module | Namespace | Source | Purpose |
|--------|-----------|--------|---------|
| **ai** | `abi.ai` | `src/features/ai/` | Full monolith -- 17+ submodules covering the entire AI surface |
| **ai_core** | `abi.ai_core` | `src/features/ai_core/` | Agents, tools, prompts, personas, memory, model discovery |
| **inference** | `abi.inference` | `src/features/ai_inference/` | LLM inference (GGUF), embeddings, vision, streaming, transformer |
| **training** | `abi.training` | `src/features/ai_training/` | Training pipelines, federated learning, data loading, WDBX bridge |
| **reasoning** | `abi.reasoning` | `src/features/ai_reasoning/` | Abbey engine, RAG, eval, templates, explore, orchestration, documents |

The `ai` module is the original monolith that re-exports submodules directly.
The four split modules (`ai_core`, `inference`, `training`, `reasoning`) provide
focused slices of the same underlying code, giving finer-grained control over
what ships in your binary.

## Build Flags

Each module is controlled by one or more build flags passed to `zig build`:

| Flag | Controls | Default |
|------|----------|---------|
| `-Denable-ai=true` | `ai`, `ai_core`, embeddings, agents, personas | `true` |
| `-Denable-llm=true` | `inference`, LLM engine | `true` |
| `-Denable-vision=true` | Vision / multimodal processing | `true` |
| `-Denable-explore=true` | Codebase exploration agent | `true` |
| `-Denable-training=true` | `training`, training pipelines, federated learning | `true` |
| `-Denable-reasoning=true` | `reasoning`, Abbey, RAG, eval, orchestration | `true` |

Disable any flag to compile out the corresponding module:

```bash
# Build with only inference, no training or reasoning
zig build -Denable-training=false -Denable-reasoning=false
```

## How Modules Relate

```
src/abi.zig (comptime feature selection)
 |
 +-- abi.ai           (monolith, -Denable-ai)
 |     |-- agent, agents, multi_agent
 |     |-- llm, embeddings, vision
 |     |-- training, federated, database
 |     |-- abbey, rag, eval, templates
 |     |-- explore, orchestration, documents
 |     |-- personas, prompts, memory
 |     |-- streaming, transformer
 |     +-- tools, discovery, gpu_agent
 |
 +-- abi.ai_core      (agents slice, -Denable-ai)
 |     |-- agent, agents, multi_agent
 |     |-- tools, prompts, memory
 |     |-- models, model_registry
 |     +-- gpu_agent, discovery
 |
 +-- abi.inference     (inference slice, -Denable-llm)
 |     |-- llm, embeddings, vision
 |     |-- streaming, transformer
 |     +-- personas
 |
 +-- abi.training      (training slice, -Denable-training)
 |     |-- training (pipelines, checkpoints, data loading)
 |     |-- federated
 |     +-- database (WDBX token datasets)
 |
 +-- abi.reasoning     (reasoning slice, -Denable-reasoning)
       |-- abbey (engine, neural, memory, calibration)
       |-- rag (chunking, retrieval, context building)
       |-- eval (evaluation framework)
       |-- templates (prompt templates)
       |-- explore (codebase exploration)
       |-- orchestration (multi-model routing)
       +-- documents (document understanding)
```

The split modules import from the shared `src/features/ai/` subdirectories.
They do not depend on each other -- you can enable `inference` without
`training`, or `reasoning` without `ai_core`.

## Quick Start

```zig
const abi = @import("abi");

// Initialize framework with default AI settings
var fw = try abi.Framework.init(allocator, .{
    .ai = .{
        .llm = .{ .model_path = "./models/llama-7b.gguf" },
        .embeddings = .{ .dimension = 768 },
    },
});
defer fw.deinit();

// Access AI context through the monolith
const ai_ctx = try fw.getAi();

// Or use the focused modules directly
const core_ctx = try abi.ai_core.Context.init(allocator, ai_config);
defer core_ctx.deinit();
```

## CLI Commands

The following CLI commands interact with the AI subsystem:

| Command | Description |
|---------|-------------|
| `abi agent` | Run AI agent (interactive or one-shot) |
| `abi llm` | LLM inference (info, generate, chat, bench, download) |
| `abi embed` | Generate embeddings (openai, mistral, cohere, ollama) |
| `abi train` | Training pipeline (run, resume, info) |

```bash
zig build run -- llm chat -m ./models/llama.gguf
zig build run -- agent --persona abbey
zig build run -- train run --config training.json
zig build run -- embed --provider ollama "Hello world"
```

## Checking Feature Status

At runtime, each module exposes an `isEnabled()` function:

```zig
if (abi.ai_core.isEnabled()) {
    // AI core is compiled in
}

if (abi.inference.isEnabled()) {
    // LLM inference is available
}

if (abi.training.isEnabled()) {
    // Training pipelines are available
}

if (abi.reasoning.isEnabled()) {
    // Abbey, RAG, eval, orchestration are available
}
```

## Related Pages

- [AI Core](ai-core.html) -- Agents, tools, prompts, personas, memory
- [Inference](ai-inference.html) -- LLM, embeddings, vision, streaming
- [Training](ai-training.html) -- Training pipelines, federated learning
- [Reasoning](ai-reasoning.html) -- Abbey, RAG, eval, orchestration

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
