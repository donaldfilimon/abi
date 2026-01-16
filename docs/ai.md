# AI & Agents

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

The training pipeline writes checkpoints using the `abi.ai.training.checkpoint` module. Each checkpoint stores model weights, optimizer state, and a simple serialization header. Checkpoints can be re‑loaded with `abi.ai.training.loadCheckpoint` to resume training or for inference.

### CLI Usage

The `train` subcommand mirrors the API above and provides convenient flags:

```bash
zig build run -- train \
    --epochs 5 \
    --batch-size 16 \
    --model-size 256 \
    --learning-rate 0.01 \
    --optimizer sgd \
    --checkpoint-path ./mymodel.ckpt \
    --checkpoint-interval 2
```

Flags correspond to the fields of `abi.ai.training.TrainingConfig`. Use `--help` for a full list. The command prints a summary report (`TrainingReport`) on completion.

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
```

---

## See Also

- [Explore](explore.md) - Codebase exploration with AI
- [Framework](framework.md) - Configuration options
- [Compute Engine](compute.md) - Task execution for AI workloads
- [Troubleshooting](troubleshooting.md) - Common issues
*See [../TODO.md](../TODO.md) and [../ROADMAP.md](../ROADMAP.md) for the Llama‑CPP parity task list and upcoming milestones.*
