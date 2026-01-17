# ai API Reference

**Source:** `src/features/ai/mod.zig`

AI feature module with agents, transformers, training, and federated learning.

Provides high-level interfaces for AI functionality including:
- Agent creation and conversation management
- Transformer models and inference
- Training pipelines
- Federated learning coordination
- **Structured error context for debugging** (new in 2026.01)

## Agent API

```zig
const agent = @import("src/features/ai/agent.zig");

// Create an agent with configuration
var my_agent = try agent.Agent.init(allocator, .{
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

## Error Context (New)

Structured error context for AI operations:

```zig
const agent = @import("src/features/ai/agent.zig");

// Create error context for API errors
const ctx = agent.ErrorContext.apiError(
    agent.AgentError.HttpRequestFailed,
    .openai,
    "https://api.openai.com/v1/chat/completions",
    500,
    "gpt-4",
);

// Log with full context
ctx.log();

// Or format to string
const msg = try ctx.formatToString(allocator);
defer allocator.free(msg);
```

### Error Context Types

- `apiError()` - For HTTP/API errors with status codes
- `configError()` - For configuration validation errors
- `generationError()` - For response generation failures
- `retryError()` - For retry-related errors with attempt tracking

## Supported Backends

| Backend | Description |
|---------|-------------|
| `echo` | Local echo for testing |
| `openai` | OpenAI API (GPT-4, etc.) |
| `ollama` | Local Ollama instance |
| `huggingface` | HuggingFace Inference API |
| `local` | Embedded transformer model |
