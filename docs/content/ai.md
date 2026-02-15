---
title: AI & LLM
description: Agents, inference, training, reasoning, and streaming
section: Modules
order: 6
---

# AI & LLM

The ABI Framework provides a comprehensive AI stack split into five independent modules.
Each module is gated by a compile-time feature flag and follows the `mod.zig` / `stub.zig`
pattern so that disabled modules incur zero binary overhead.

## Module Overview

| Module | Namespace | Build Flag | Description |
|--------|-----------|------------|-------------|
| **ai** | `abi.ai` | `-Denable-ai` | Full monolith with 17+ submodules (backward-compatible) |
| **ai_core** | `abi.ai_core` | `-Denable-ai` | Agents, tools, prompts, personas, memory, model discovery |
| **inference** | `abi.inference` | `-Denable-llm` | LLM engine, embeddings, vision, streaming, transformer |
| **training** | `abi.training` | `-Denable-training` | Training pipelines, federated learning, WDBX bridge |
| **reasoning** | `abi.reasoning` | `-Denable-reasoning` | Abbey engine, RAG, eval, templates, orchestration |

The monolith `abi.ai` module re-exports everything from the other four for backward
compatibility. New code should prefer the split modules for finer-grained control.

## Build Flags

```bash
# Enable individual AI subsystems
zig build -Denable-ai=true       # ai + ai_core + embeddings + agents
zig build -Denable-llm=true      # LLM inference engine
zig build -Denable-vision=true   # Vision processing
zig build -Denable-training=true # Training pipelines
zig build -Denable-reasoning=true # Abbey, RAG, eval, orchestration
zig build -Denable-explore=true  # Codebase exploration agent
```

All flags default to `true` (enabled). Disable with `=false` to strip the
corresponding code from the binary.

## AI Core -- Agents, Tools, Prompts

The `ai_core` module (`src/features/ai_core/mod.zig`) provides the foundational
building blocks for AI applications.

### Agent System

The `Agent` struct supports conversational interactions with configurable backends,
history management, and sampling parameters.

```zig
const abi = @import("abi");

var agent = try abi.ai_core.Agent.init(allocator, .{
    .name = "assistant",
    .backend = .echo,           // .echo, .openai, .ollama, .huggingface, .local
    .enable_history = true,
    .temperature = 0.7,
    .top_p = 0.9,
    .max_tokens = 1024,
});
defer agent.deinit();
```

**AgentBackend options:**

| Backend | Description |
|---------|-------------|
| `.echo` | Local echo for testing and fallback |
| `.openai` | OpenAI Chat Completions API |
| `.ollama` | Local Ollama inference server |
| `.huggingface` | HuggingFace Inference API |
| `.local` | Local transformer model |

### Tool Registry

Tools let agents invoke structured operations. Register tools, then attach them
to agents via the `ToolAgent` wrapper.

```zig
const tools = abi.ai_core.tools;

var registry = tools.ToolRegistry.init(allocator);
defer registry.deinit();

try registry.register(.{
    .name = "search",
    .description = "Search the knowledge base",
    .handler = mySearchHandler,
});
```

Built-in tool sets include `OsTools` (file system, process), `DiscordTools`
(bot messaging), `TaskTool`, and `Subagent`.

### Prompt Builder and Personas

Build structured prompts with system instructions, few-shot examples, and
persona configuration.

```zig
var prompt = abi.ai_core.PromptBuilder.init(allocator);
defer prompt.deinit();

try prompt.system("You are a helpful assistant.");
try prompt.user("Explain comptime in Zig.");
```

The `Persona` type defines reusable personality profiles (`PersonaType`) that
shape agent behavior across conversations.

### Multi-Agent Coordination

The `MultiAgentCoordinator` orchestrates multiple agents working on a shared
task, routing messages between them and managing turn-taking.

### Model Discovery

`ModelDiscovery` scans the system for available models and adapts configuration
based on `SystemCapabilities` (memory, compute, GPU presence).

## Inference -- LLM, Embeddings, Vision, Streaming

The `inference` module (`src/features/ai_inference/mod.zig`) handles all
inference-time operations.

### LLM Engine

Load and run GGUF models locally with the `LlmEngine`:

```zig
const inference = abi.inference;

var engine = try inference.LlmEngine.init(allocator, .{
    .model_path = "./models/llama-7b.gguf",
});
defer engine.deinit();
```

Key types: `LlmEngine`, `LlmModel`, `LlmConfig`, `GgufFile`, `BpeTokenizer`.

### Transformer

The transformer module provides the core `TransformerModel` and
`TransformerConfig` for local model execution with attention, feed-forward
layers, and KV caching.

### Streaming Server

The streaming module provides production-ready HTTP servers with SSE and
WebSocket support:

```zig
const streaming = abi.inference.streaming;

var server = try streaming.StreamingServer.init(allocator, .{
    .address = "0.0.0.0:8080",
    .auth_token = "secret-token",
    .default_model_path = "./models/llama-7b.gguf",
    .preload_model = true,
    .enable_recovery = true,
});
defer server.deinit();

try server.serve(); // Blocking
```

**CLI shortcut:**

```bash
zig build run -- llm serve -m model.gguf
# or the alias:
zig build run -- serve -m model.gguf
```

**Streaming features:**

- SSE (Server-Sent Events) and WebSocket endpoints
- OpenAI-compatible `/v1/chat/completions` API
- Backend routing: local GGUF, OpenAI, Ollama, Anthropic
- Backpressure control to prevent memory exhaustion
- Circuit breakers for per-backend failure isolation
- Retry logic with exponential backoff and jitter
- Session caching for reconnection recovery via `Last-Event-ID`

| Error Type | Handling Strategy |
|------------|-------------------|
| Connection errors | Retry with exponential backoff |
| Backend failures | Circuit breaker isolation + fallback |
| Rate limiting | Backpressure + client notification |
| Stream interruption | Session cache for resumption |
| Timeout | Configurable per-token and total timeouts |

### Embeddings and Vision

- **Embeddings**: Generate vector embeddings for similarity search (gated by `-Denable-ai`)
- **Vision**: Image processing and analysis (gated by `-Denable-vision`)

## Training -- Pipelines and Federated Learning

The `training` module (`src/features/ai_training/mod.zig`) provides model
training infrastructure.

### Training Pipelines

```zig
const training = abi.training;

var config = training.TrainingConfig{
    .learning_rate = 1e-4,
    .batch_size = 32,
    .epochs = 10,
    .optimizer = .adam,
    .schedule = .cosine,
};
```

Key types: `TrainingConfig`, `TrainingReport`, `TrainingResult`,
`OptimizerType` (SGD, Adam), `LearningRateSchedule`, `CheckpointStore`.

**CLI:**

```bash
zig build run -- train run                     # Run training pipeline
zig build run -- train generate-data           # Generate synthetic token data
zig build run -- train --external-quantize     # Shell to llama-quantize
```

### Trainable Models

- `TrainableModel` / `LlamaTrainer` -- LLM fine-tuning
- `TrainableViTModel` -- Vision Transformer training
- `TrainableCLIPModel` -- Multimodal CLIP training

### Checkpointing

Save and restore training state:

```zig
try training.saveCheckpoint(store, model, epoch);
const restored = try training.loadCheckpoint(store, path);
```

### Data Loading

`TokenizedDataset`, `DataLoader`, and `BatchIterator` provide efficient
data pipelines. `SequencePacker` handles variable-length sequence packing
for transformer training.

### WDBX Database Bridge

The training module bridges to the WDBX vector database for storing and
loading tokenized datasets:

```zig
try training.tokenBinToWdbx(allocator, "tokens.bin", db_handle);
try training.wdbxToTokenBin(allocator, db_handle, "output.bin");
```

### Federated Learning

The `federated` submodule supports distributed training across multiple
nodes with gradient aggregation and privacy-preserving updates.

## Reasoning -- Abbey, RAG, Eval

The `reasoning` module (`src/features/ai_reasoning/mod.zig`) provides
advanced AI reasoning capabilities.

### Abbey Engine

Abbey is a comprehensive, emotionally intelligent AI framework with:

- **Neural learning and attention**: Multi-head attention mechanisms
  with configurable layers
- **Three-tier memory**: Episodic (events), semantic (facts), working
  (active context)
- **Confidence calibration**: Bayesian updating for prediction
  confidence
- **Emotional intelligence**: Adaptive responses based on emotional
  state tracking
- **Meta-learning and self-reflection**: The engine can reason about
  its own reasoning process
- **Theory of mind**: Models the user's mental state for more
  effective communication

Key types: `AbbeyEngine`, `Abbey`, `ReasoningChain`, `ReasoningStep`,
`ConversationContext`.

### RAG (Retrieval-Augmented Generation)

The `rag` submodule retrieves relevant documents from a vector store
and injects them into the prompt context before generation.

### Eval Framework

The `eval` submodule provides evaluation metrics and benchmarks for
measuring model quality, including automated scoring and human-in-the-loop
evaluation flows.

### Templates

Reusable prompt templates with variable substitution, conditional
sections, and composition.

### Orchestration

The `Orchestrator` routes tasks to the best-suited model or pipeline
based on `TaskType`, with support for ensemble methods and routing
strategies:

```zig
const orchestration = abi.reasoning.orchestration;

var orch = try orchestration.Orchestrator.init(allocator, .{
    .strategy = .cost_optimized,
});
defer orch.deinit();
```

Routing strategies include cost-optimized, latency-optimized, and
quality-optimized modes. `EnsembleMethod` supports majority voting,
weighted averaging, and cascading fallback.

### Codebase Exploration

The `explore` submodule (gated by `-Denable-explore`) provides an
`ExploreAgent` that indexes and searches codebases with query
understanding (`QueryIntent`, `ParsedQuery`) for natural-language
code questions.

### Document Understanding

The `documents` submodule processes and understands structured documents
for use in RAG pipelines and knowledge extraction workflows.

## Quick Start

```zig
const abi = @import("abi");

// Initialize framework with AI features
var fw = try abi.Framework.init(allocator, .{
    .ai = .{
        .llm = .{ .model_path = "./models/llama-7b.gguf" },
        .embeddings = .{ .dimension = 768 },
    },
});
defer fw.deinit();

// Access AI context
const ai_ctx = try fw.getAi();

// Check sub-feature availability
if (ai_ctx.isSubFeatureEnabled(.llm)) {
    const llm = try ai_ctx.getLlm();
    // ... perform inference ...
}
```

## Related

- [Connectors](connectors.md) -- 9 LLM provider integrations
- [GPU](gpu.md) -- Hardware acceleration for inference and training
- [Database](database.md) -- WDBX vector database for embeddings
