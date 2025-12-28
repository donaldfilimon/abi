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

## Agents

An **Agent** wraps a connector with memory and tools.

```zig
var agent = try abi.ai.Agent.init(allocator, connector, .{
    .system_prompt = "You are a helpful coding assistant.",
});
defer agent.deinit();

const response = try agent.chat("How do I write a Hello World in Zig?");
```
