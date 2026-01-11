# AI & Agents

The **AI** module (`abi.ai`) provides the building blocks for creating autonomous agents and connecting to LLM providers.

## Connectors

Connectors provide a unified interface to various model providers.

Supported (or planned) providers:

- **OpenAI** (`abi.connectors.openai`)
- **Ollama** (`abi.connectors.ollama`)
- **HuggingFace** (`abi.connectors.huggingface`)

### Configuration

Connectors are typically configured via environment variables for security.

- `ABI_OPENAI_API_KEY`
- `ABI_OLLAMA_HOST`

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
});
```

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

An **Agent** wraps a connector with memory and tools.

```zig
var agent = try abi.ai.Agent.init(allocator, .{
    .name = "coding-assistant",
    .system_prompt = "You are a helpful coding assistant.",
    .backend = .ollama,
    .enable_history = true,
    .temperature = 0.7,
});
defer agent.deinit();

const response = try agent.chat("How do I write a Hello World in Zig?");
```

### Backend Options

| Backend | Description |
|---------|-------------|
| `.echo` | Local echo for testing/fallback |
| `.openai` | OpenAI API (requires `ABI_OPENAI_API_KEY`) |
| `.ollama` | Ollama local models (default: `http://127.0.0.1:11434`) |
| `.huggingface` | HuggingFace Inference API |
| `.local` | Local scheduler |
