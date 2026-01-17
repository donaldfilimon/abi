# AI & Agents

> **Developer Guide**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for coding patterns and [CLAUDE.md](../CLAUDE.md) for comprehensive agent guidance.
> **Framework**: Initialize ABI framework before using AI features - see [Framework Guide](framework.md).

The **AI** module (`abi.ai`) provides the building blocks for creating autonomous agents and connecting to LLM providers.

## Connectors

Connectors provide a unified interface to various model providers and platforms.

### Model Providers

- **OpenAI** (`abi.connectors.openai`) - GPT-4, GPT-3.5, embeddings
- **Ollama** (`abi.connectors.ollama`) - Local LLM inference
- **HuggingFace** (`abi.connectors.huggingface`) - Inference API

### Platform Integrations

- **Discord** (`abi.connectors.discord`) - Discord Bot API for messaging, webhooks, and interactions

### Configuration

Connectors are typically configured via environment variables for security.

- `ABI_OPENAI_API_KEY` - OpenAI API key
- `ABI_OLLAMA_HOST` - Ollama server URL (default: `http://127.0.0.1:11434`)
- `DISCORD_BOT_TOKEN` - Discord bot authentication token

## Training
The training pipeline (`abi.ai.training`) supports gradient accumulation and
checkpointing with a simple simulation backend.

```zig
const report = try abi.ai.train(allocator, .{
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

The training pipeline writes weight-only checkpoints using the `abi.ai.training.checkpoint` module. LLM training uses `abi.ai.training.llm_checkpoint` to persist model weights and optimizer state together. Checkpoints can be re‑loaded with `abi.ai.training.loadCheckpoint` (generic) or `abi.ai.training.loadLlmCheckpoint` (LLM) to resume training or for inference.

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
Federated coordination (`abi.ai.federated`) aggregates model updates across nodes.

```zig
var coordinator = try abi.ai.federated.Coordinator.init(allocator, .{}, 128);
defer coordinator.deinit();

try coordinator.registerNode("node-a");
try coordinator.submitUpdate(.{
    .node_id = "node-a",
    .step = 1,
    .weights = &.{ 0.1, 0.2 },
});
const global = try coordinator.aggregate();
```

## Agents

An **Agent** provides a conversational interface with configurable history and parameters.

```zig
var agent = try abi.ai.Agent.init(allocator, .{
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

## See Also

- [Explore](explore.md) - Codebase exploration with AI
- [Framework](framework.md) - Configuration options
- [Compute Engine](compute.md) - Task execution for AI workloads
- [Troubleshooting](troubleshooting.md) - Common issues
*See [../TODO.md](../TODO.md) and [../ROADMAP.md](../ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
