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
