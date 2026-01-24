---
title: "ai"
tags: []
---
# AI & Agents
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Module-AI-purple?style=for-the-badge&logo=openai&logoColor=white" alt="AI Module"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge" alt="Production Ready"/>
  <img src="https://img.shields.io/badge/LLM-Llama_CPP_Parity-blue?style=for-the-badge" alt="Llama CPP Parity"/>
</p>

<p align="center">
  <a href="#connectors">Connectors</a> •
  <a href="#llm-sub-feature">LLM</a> •
  <a href="#agents-sub-feature">Agents</a> •
  <a href="#training-sub-feature">Training</a> •
  <a href="#cli-commands">CLI</a>
</p>

---

> **Developer Guide**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for coding patterns and [CLAUDE.md](../CLAUDE.md) for comprehensive agent guidance.
> **Framework**: Initialize ABI framework before using AI features - see [Framework Guide](framework.md).

The **AI** module (`abi.ai`) provides the building blocks for creating autonomous agents and connecting to LLM providers.

## Sub-Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| **LLM** | Local LLM inference (GGUF support) | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Embeddings** | Vector embedding generation | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Agents** | Conversational AI agents | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Training** | Training pipelines & checkpointing | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Connectors** | OpenAI, Ollama, HuggingFace | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **GPU-Aware Agent** | RL-based GPU scheduling for AI workloads | ![Ready](https://img.shields.io/badge/-Ready-success) |

## Architecture

The AI module uses a modular architecture with a core module and independent sub-features:

```
src/ai/
├── mod.zig              # AI module entry point (core)
├── core/                # Core AI primitives
│   ├── embeddings.zig   # Embedding generation
│   ├── inference.zig    # Inference engine
│   └── tokenizer.zig    # Tokenization
├── llm/                 # LLM sub-feature
│   ├── mod.zig          # LLM entry point
│   ├── gguf.zig         # GGUF model loading
│   └── quantization.zig # Quantization support
├── embeddings/          # Embeddings sub-feature
│   ├── mod.zig          # Embeddings entry point
│   └── models/          # Embedding models
├── agents/              # Agents sub-feature
│   ├── mod.zig          # Agent entry point
│   ├── agent.zig        # Agent implementation
│   └── prompts/         # Prompt templates
├── training/            # Training sub-feature
│   ├── mod.zig          # Training entry point
│   ├── trainer.zig      # Training loop
│   ├── checkpoint.zig   # Checkpointing
│   └── federated.zig    # Federated learning
└── connectors/          # External provider connectors
    ├── openai.zig       # OpenAI API
    ├── ollama.zig       # Ollama local inference
    ├── huggingface.zig  # HuggingFace API
    └── discord.zig      # Discord Bot API
```

Each sub-feature (llm, embeddings, agents, training) can be independently enabled or disabled, and they share the core primitives.

## Connectors

Connectors provide a unified interface to various model providers and platforms.

### Model Providers

| Provider | Namespace | Models | Status |
|----------|-----------|--------|--------|
| **OpenAI** | `abi.ai.connectors.openai` | GPT-4, GPT-3.5, embeddings | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **Ollama** | `abi.ai.connectors.ollama` | Local LLMs (Llama, Mistral, etc.) | ![Ready](https://img.shields.io/badge/-Ready-success) |
| **HuggingFace** | `abi.ai.connectors.huggingface` | Inference API models | ![Ready](https://img.shields.io/badge/-Ready-success) |

### Platform Integrations

| Platform | Namespace | Features | Status |
|----------|-----------|----------|--------|
| **Discord** | `abi.ai.connectors.discord` | Bot API, webhooks, interactions | ![Ready](https://img.shields.io/badge/-Ready-success) |

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ABI_OPENAI_API_KEY` | - | OpenAI API key |
| `ABI_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `ABI_OLLAMA_MODEL` | `gpt-oss` | Default Ollama model |
| `ABI_HF_API_TOKEN` | - | HuggingFace token |
| `DISCORD_BOT_TOKEN` | - | Discord bot token |

## LLM Sub-Feature

The LLM sub-feature (`abi.ai.llm`) provides local LLM inference capabilities with both batch and streaming generation.

### Basic Generation

```zig
const llm = abi.ai.llm;

// Load a GGUF model
var model = try llm.loadModel(allocator, "model.gguf", .{});
defer model.deinit();

// Generate text
const output = try model.generate("Hello, ", .{
    .max_tokens = 100,
    .temperature = 0.7,
});
defer allocator.free(output);
```

### Streaming Generation

The LLM module provides advanced streaming capabilities for token-by-token generation, ideal for interactive applications and real-time output.

#### Callback-based Streaming

Simple callback-based streaming for quick implementation:

```zig
var engine = llm.Engine.init(allocator, .{});
defer engine.deinit();

try engine.loadModel("model.gguf");

// Stream with callback
try engine.generateStreaming("Once upon a time", struct {
    fn onToken(text: []const u8) void {
        std.debug.print("{s}", .{text});
    }
}.onToken);
```

#### Iterator-based Streaming with StreamingResponse

For more control, use the `StreamingResponse` iterator which provides pull-based streaming with statistics and cancellation support:

```zig
const llm = abi.ai.llm;

var engine = llm.Engine.init(allocator, .{});
defer engine.deinit();

try engine.loadModel("model.gguf");

// Create streaming response with configuration
var response = try engine.createStreamingResponse("Write a story about", .{
    .max_tokens = 200,
    .temperature = 0.8,
    .top_k = 40,
    .top_p = 0.9,
    .decode_tokens = true,
});
defer response.deinit();

// Iterate through tokens
while (try response.next()) |event| {
    if (event.text) |text| {
        try stdout.writeAll(text);
    }
    if (event.is_final) break;
}

// Get generation statistics
const stats = response.getStats();
std.debug.print("\n\nTokens: {d}, Speed: {d:.1} tok/s\n", .{
    stats.tokens_generated,
    stats.tokensPerSecond(),
});
```

#### Streaming with Callbacks and Configuration

Combine callbacks with the advanced configuration:

```zig
const stats = try engine.generateStreamingWithConfig("Hello, ", .{
    .max_tokens = 100,
    .temperature = 0.7,
    .on_token = struct {
        fn onToken(event: llm.TokenEvent) void {
            if (event.text) |text| {
                std.debug.print("{s}", .{text});
            }
        }
    }.onToken,
    .on_complete = struct {
        fn onComplete(stats: llm.StreamingStats) void {
            std.debug.print("\nGenerated {d} tokens\n", .{stats.tokens_generated});
        }
    }.onComplete,
});
```

### Streaming Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_tokens` | `u32` | 256 | Maximum tokens to generate |
| `temperature` | `f32` | 0.7 | Sampling temperature (0.0 = greedy) |
| `top_k` | `u32` | 40 | Top-k sampling (0 = disabled) |
| `top_p` | `f32` | 0.9 | Nucleus sampling threshold |
| `repetition_penalty` | `f32` | 1.1 | Repetition penalty (1.0 = disabled) |
| `stop_tokens` | `[]const u32` | EOS | Stop token IDs |
| `initial_buffer_capacity` | `u32` | 256 | Token buffer initial size |
| `decode_tokens` | `bool` | true | Decode tokens to text |
| `generation_timeout_ns` | `u64` | 0 | Generation timeout (0 = none) |
| `on_token` | callback | null | Per-token callback |
| `on_complete` | callback | null | Completion callback |

### Streaming Types

- `StreamingResponse` - Iterator for pull-based streaming
- `StreamingConfig` - Configuration for streaming behavior
- `TokenEvent` - Event emitted for each token (includes text, position, timestamps)
- `StreamingStats` - Generation statistics (tokens/sec, time-to-first-token)
- `StreamingState` - Current state (idle, prefilling, generating, completed, cancelled)

### Server-Sent Events (SSE) Support

Built-in SSE formatting for web APIs:

```zig
const llm = abi.ai.llm;

// Format token as SSE
const sse_data = try llm.SSEFormatter.formatTokenEvent(allocator, event);
defer allocator.free(sse_data);
// Output: data: {"token_id":123,"text":"hello","position":5,"is_final":false}\n\n

// Format completion as SSE
const completion_sse = try llm.SSEFormatter.formatCompletionEvent(allocator, stats);
defer allocator.free(completion_sse);
// Output: data: {"event":"complete","tokens_generated":50,"tokens_per_second":25.0}\n\n
```

### Cancellation Support

```zig
var response = try engine.createStreamingResponse(prompt, config);
defer response.deinit();

// In another thread or from a signal handler
response.cancel();

// In the main loop, cancelled responses return null
while (try response.next()) |event| {
    // Process event...
}
// Loop exits when cancelled

if (response.isCancelled()) {
    std.debug.print("Generation was cancelled\n", .{});
}
```

## Embeddings Sub-Feature

The embeddings sub-feature (`abi.ai.embeddings`) generates vector embeddings.

```zig
const embeddings = abi.ai.embeddings;

var encoder = try embeddings.Encoder.init(allocator, .{});
defer encoder.deinit();

const vector = try encoder.encode("Hello, world!");
defer allocator.free(vector);
```

## Training Sub-Feature

The training sub-feature (`abi.ai.training`) supports gradient accumulation and checkpointing with a simple simulation backend.

```zig
const report = try abi.ai.training.train(allocator, .{
    .epochs = 2,
    .batch_size = 8,
    .sample_count = 64,
    .model_size = 128,
    .gradient_accumulation_steps = 2,
    .checkpoint_interval = 1,
    .max_checkpoints = 3,
    .checkpoint_path = "./model.ckpt",
});
```

The training pipeline writes weight-only checkpoints using the `abi.ai.training.checkpoint` module. LLM training uses `abi.ai.training.llm_checkpoint` to persist model weights and optimizer state together. Checkpoints can be re-loaded with `abi.ai.training.loadCheckpoint` (generic) or `abi.ai.training.loadLlmCheckpoint` (LLM) to resume training or for inference.

### LLM Training Extras

The LLM trainer supports training metrics, checkpointing with optimizer state, TensorBoard/W&B logging, and GGUF export.

Key options on `LlmTrainingConfig`:
- `checkpoint_interval`, `checkpoint_path` - Save LLM checkpoints (weights + optimizer state)
- `log_dir`, `enable_tensorboard`, `enable_wandb` - Scalar metrics logging
- `export_gguf_path` - Export trained weights to GGUF after training

W&B logging writes offline run files under `log_dir/wandb/` (sync with `wandb sync` if desired).

### CLI Usage

The `train` command provides subcommands for running and managing training:

```bash
# Run training with default configuration
zig build run -- train run

# Run with custom options
zig build run -- train run --epochs 5 --batch-size 16 --learning-rate 0.01

# Run with optimizer and checkpointing
zig build run -- train run \
    -e 10 \
    -b 32 \
    --model-size 512 \
    --optimizer adamw \
    --lr-schedule warmup_cosine \
    --checkpoint-interval 100 \
    --checkpoint-path ./checkpoints/model.ckpt

# Show default configuration
zig build run -- train info

# Resume from checkpoint
zig build run -- train resume ./checkpoints/model.ckpt

# Show all options
zig build run -- train help
```

**Available options:**
- `-e, --epochs` - Number of epochs (default: 10)
- `-b, --batch-size` - Batch size (default: 32)
- `--model-size` - Model parameters (default: 512)
- `--lr, --learning-rate` - Learning rate (default: 0.001)
- `--optimizer` - sgd, adam, adamw (default: adamw)
- `--lr-schedule` - constant, cosine, warmup_cosine, step, polynomial
- `--checkpoint-interval` - Steps between checkpoints
- `--checkpoint-path` - Path to save checkpoints
- `--mixed-precision` - Enable mixed precision training

See `src/tests/training_demo.zig` for a working test example.

## Federated Learning

Federated coordination (`abi.ai.training.federated`) aggregates model updates across nodes.

```zig
var coordinator = try abi.ai.training.federated.Coordinator.init(allocator, .{}, 128);
defer coordinator.deinit();

try coordinator.registerNode("node-a");
try coordinator.submitUpdate(.{
    .node_id = "node-a",
    .step = 1,
    .weights = &.{ 0.1, 0.2 },
});
const global = try coordinator.aggregate();
```

## Agents Sub-Feature

An **Agent** (`abi.ai.agents`) provides a conversational interface with configurable history and parameters.

```zig
var agent = try abi.ai.agents.Agent.init(allocator, .{
    .name = "coding-assistant",
    .enable_history = true,
    .temperature = 0.7,
    .top_p = 0.9,
});
defer agent.deinit();

// Use chat() for conversational interface
const response = try agent.chat("How do I write a Hello World in Zig?", allocator);
defer allocator.free(response);

// Or use process() for the same functionality
const response2 = try agent.process("Another question", allocator);
defer allocator.free(response2);
```

### Agent Configuration

The `AgentConfig` struct supports:
- `name: []const u8` - Agent identifier (required)
- `enable_history: bool` - Enable conversation history (default: true)
- `temperature: f32` - Sampling temperature 0.0-2.0 (default: 0.7)
- `top_p: f32` - Nucleus sampling parameter 0.0-1.0 (default: 0.9)

### Agent Methods

- `chat(input, allocator)` - Process input and return response (conversational interface)
- `process(input, allocator)` - Same as chat(), alternative naming
- `historyCount()` - Get number of history entries
- `historySlice()` - Get conversation history
- `clearHistory()` - Clear conversation history
- `setTemperature(temp)` - Update temperature
- `setTopP(top_p)` - Update top_p parameter
- `setHistoryEnabled(enabled)` - Enable/disable history tracking

### GPU-Aware Agent

The **GPU-Aware Agent** (`abi.ai.GpuAgent`) integrates AI capabilities with intelligent GPU scheduling, using reinforcement learning to optimize resource allocation.

```zig
const abi = @import("abi");

var agent = try abi.ai.GpuAgent.init(allocator);
defer agent.deinit();

// Process request with GPU-aware scheduling
const response = try agent.process(.{
    .prompt = "Analyze this dataset...",
    .workload_type = .inference,
    .priority = .high,
    .max_tokens = 2048,
    .memory_hint_mb = 4096,
});
defer allocator.free(response.content);

std.debug.print("Backend: {s}, Latency: {d}ms\n", .{
    response.gpu_backend_used,
    response.latency_ms,
});

// Get learning statistics
if (agent.getGpuStats()) |stats| {
    std.debug.print("Episodes: {d}, Avg Reward: {d:.2}\n", .{
        stats.episodes,
        stats.avg_episode_reward,
    });
}
```

**Workload Types:**
- `.inference` - Standard LLM inference
- `.training` - Model training (GPU-intensive)
- `.embedding` - Vector embedding generation
- `.fine_tuning` - Model fine-tuning
- `.batch_inference` - Batch processing

**Features:**
- Automatic GPU backend selection via RL
- Workload profiling and memory hints
- Learning-based scheduling optimization
- Integration with Mega GPU coordinator

---

## Multi-Persona System

The **Multi-Persona AI Assistant** (`abi.ai.personas`) provides intelligent routing between specialized AI personas:

| Persona | Role | Characteristics |
|---------|------|-----------------|
| **Abi** | Router/Moderator | Content moderation, sentiment analysis, policy enforcement |
| **Abbey** | Empathetic Polymath | Supportive, thorough responses with emotional awareness |
| **Aviva** | Direct Expert | Concise, factual, technically rigorous responses |

### Quick Start

```zig
const personas = abi.ai.personas;

// Initialize the multi-persona system
var system = try personas.MultiPersonaSystem.init(allocator, .{
    .default_persona = .abbey,
    .enable_dynamic_routing = true,
});
defer system.deinit();

// Process a request with automatic routing
const request = personas.PersonaRequest{
    .content = "Help me understand memory management in Zig",
    .user_id = "user-123",
};

const response = try system.process(request);
defer @constCast(&response).deinit(allocator);

// Response includes which persona handled it
std.debug.print("Persona: {s}, Content: {s}\n", .{
    @tagName(response.persona),
    response.content,
});
```

### Persona Configuration

```zig
const cfg = personas.MultiPersonaConfig{
    .default_persona = .abbey,
    .enable_dynamic_routing = true,
    .routing_confidence_threshold = 0.5,
    .abbey = .{
        .empathy_level = .high,
        .response_depth = .thorough,
    },
    .aviva = .{
        .cite_sources = true,
        .skip_preamble = true,
    },
};
```

### Routing Logic

The system routes requests based on:

1. **Sentiment Analysis** - Detects emotional tone and urgency
2. **Policy Checking** - Ensures content compliance
3. **Query Classification** - Identifies request type (code, factual, explanation)
4. **Rules Engine** - Applies routing rules based on analysis

**Example routing scenarios:**
- Frustrated user → Abbey (empathetic support)
- Technical code request → Aviva (direct expertise)
- Policy violation → Abi (moderation)

### HTTP API

The personas module provides HTTP API handlers:

```
POST /api/v1/chat              # Auto-routing to best persona
POST /api/v1/chat/abbey        # Force Abbey persona
POST /api/v1/chat/aviva        # Force Aviva persona
GET  /api/v1/personas          # List available personas
GET  /api/v1/personas/metrics  # Get persona metrics
GET  /api/v1/personas/health   # Health check
```

**Request format:**
```json
{
  "content": "Help me understand recursion",
  "user_id": "user-123",
  "session_id": "session-456",
  "persona": null
}
```

**Response format:**
```json
{
  "content": "Recursion is when a function calls itself...",
  "persona": "abbey",
  "confidence": 0.92,
  "latency_ms": 450
}
```

### Metrics & Monitoring

```zig
// Access persona metrics
if (system.metrics) |m| {
    const stats = m.getStats(.abbey);
    if (stats) |s| {
        std.debug.print("Abbey: {d} requests, {d:.1}% success\n", .{
            s.total_requests,
            s.success_rate * 100,
        });
    }
}

// Get latency percentiles
if (m.getLatencyPercentiles(.abbey)) |lat| {
    std.debug.print("P50: {d:.0}ms, P99: {d:.0}ms\n", .{
        lat.p50, lat.p99,
    });
}
```

### Architecture

```
src/ai/personas/
├── mod.zig              # Main orchestrator
├── types.zig            # Core types (PersonaRequest, PersonaResponse)
├── config.zig           # Configuration structs
├── registry.zig         # Persona registry
├── metrics.zig          # Metrics with percentile tracking
├── loadbalancer.zig     # Health-weighted load balancing
├── health.zig           # Health checking
├── alerts.zig           # Alert rules and manager
├── abi/                 # Router persona
│   ├── sentiment.zig    # Sentiment analysis
│   ├── policy.zig       # Content moderation
│   └── rules.zig        # Routing rules engine
├── abbey/               # Empathetic persona
│   ├── emotion.zig      # Emotion detection
│   ├── empathy.zig      # Empathy injection
│   └── reasoning.zig    # Reasoning chains
└── aviva/               # Expert persona
    ├── classifier.zig   # Query classification
    ├── knowledge.zig    # Knowledge retrieval
    ├── code.zig         # Code generation
    └── facts.zig        # Fact checking
```

For detailed API documentation, see [Personas API Reference](api/personas.md).

---

## CLI Commands

```bash
# Run AI agent interactively
zig build run -- agent

# Run with a single message
zig build run -- agent --message "Hello, how are you?"

# LLM model operations
zig build run -- llm info model.gguf       # Show model information
zig build run -- llm generate model.gguf   # Generate text
zig build run -- llm chat model.gguf       # Interactive chat
zig build run -- llm bench model.gguf      # Benchmark performance

# Training pipeline
zig build run -- train run --epochs 10     # Run training
zig build run -- train info                # Show configuration
zig build run -- train resume ./model.ckpt # Resume from checkpoint

# Dataset conversion (TokenBin <-> WDBX)
zig build run -- convert dataset --input data.bin --output data.wdbx --format to-wdbx
zig build run -- convert dataset --input data.wdbx --output data.bin --format to-tokenbin
```

---

## New in 2026.01: Error Context

```zig
// Create and log error context
const ctx = agent.ErrorContext.apiError(err, .openai, endpoint, 500, "gpt-4");
ctx.log();  // Outputs: "AgentError: HttpRequestFailed during API request [backend=openai] [model=gpt-4]"
```

Factory methods: `apiError()`, `configError()`, `generationError()`, `retryError()`

New error types: `Timeout`, `ConnectionRefused`, `ModelNotFound`

---

## API Reference

**Source:** `src/ai/mod.zig`

### Agent API

```zig
const abi = @import("abi");

// Create an agent with configuration
var my_agent = try abi.ai.Agent.init(allocator, .{
    .name = "assistant",
    .backend = .openai,
    .model = "gpt-4",
    .temperature = 0.7,
    .system_prompt = "You are a helpful assistant.",
});
defer my_agent.deinit();

// Process a message
const response = try my_agent.process("Hello!", allocator);
defer allocator.free(response);
```

### Error Context Types

- `apiError()` - For HTTP/API errors with status codes
- `configError()` - For configuration validation errors
- `generationError()` - For response generation failures
- `retryError()` - For retry-related errors with attempt tracking

### Supported Backends

| Backend | Description | Use Case | Status |
|---------|-------------|----------|--------|
| `echo` | Local echo for testing | Development/testing | ![Ready](https://img.shields.io/badge/-Ready-success) |
| `openai` | OpenAI API (GPT-4, etc.) | Production cloud inference | ![Ready](https://img.shields.io/badge/-Ready-success) |
| `ollama` | Local Ollama instance | Local development | ![Ready](https://img.shields.io/badge/-Ready-success) |
| `huggingface` | HuggingFace Inference API | Model experimentation | ![Ready](https://img.shields.io/badge/-Ready-success) |
| `local` | Embedded transformer model | Offline inference | ![Ready](https://img.shields.io/badge/-Ready-success) |

---

## See Also

<table>
<tr>
<td>

### Related Guides
- [Explore](explore.md) — Codebase exploration with AI
- [Framework](framework.md) — Configuration options
- [Compute Engine](compute.md) — Task execution for AI workloads
- [GPU Acceleration](gpu.md) — GPU-accelerated inference
- [Abbey-Aviva Research](research/abbey-aviva-abi-wdbx-framework.md) — Multi-persona AI architecture

</td>
<td>

### Resources
- [Troubleshooting](troubleshooting.md) — Common issues
- [API Reference](../API_REFERENCE.md) — AI API details
- [Examples](../examples/) — Code samples

</td>
</tr>
</table>

---

<p align="center">
  <a href="docs-index.md">← Documentation Index</a> •
  <a href="gpu.md">GPU Guide →</a>
</p>
