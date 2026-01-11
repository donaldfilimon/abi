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

## Discord Integration

Discord tools are available for AI agents to interact with Discord servers.

### Available Tools

| Tool | Description |
|------|-------------|
| `discord_send_message` | Send messages to channels |
| `discord_get_channel` | Get channel information |
| `discord_list_guilds` | List connected servers |
| `discord_get_bot_info` | Get bot user details |
| `discord_execute_webhook` | Execute webhooks |
| `discord_add_reaction` | Add reactions to messages |
| `discord_get_messages` | Retrieve channel messages |

### Example

```zig
const discord_tools = @import("abi").ai.tools.discord;

// Register all Discord tools with an agent's tool registry
try discord_tools.registerAll(&agent.tool_registry);

// Tools can now be called by the agent during conversations
```

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

