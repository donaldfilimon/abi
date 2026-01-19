# AI & Agents
> **Codebase Status:** Synced with repository as of 2026-01-18.

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

The LLM sub-feature (`abi.ai.llm`) provides local LLM inference capabilities.

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
