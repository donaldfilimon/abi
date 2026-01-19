# Agents Guide
> **Codebase Status:** Synced with repository as of 2026-01-18.

> **Developer Guide**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for coding patterns and [CLAUDE.md](../CLAUDE.md) for comprehensive guidance.
>
> **Last Updated:** January 18, 2026
> **Zig Version:** 0.16.x

## Overview

The ABI Agents module (`abi.ai.agents`) provides conversational AI agents with configurable history, sampling parameters, and backend support. This guide covers agent initialization, configuration, and the Zig 0.16 environment patterns required for proper setup.

## Zig 0.16 Environment Initialization

Before using agents with file-based operations (loading configs, model files, etc.), you need to initialize the Zig 0.16 I/O environment.

### Standard Agent Initialization

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize I/O backend for file operations (Zig 0.16)
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Initialize the ABI framework
    const config = abi.Config.init()
        .withAI(true);

    var framework = try abi.Framework.init(allocator, config);
    defer framework.deinit();

    // Create an agent
    var agent = try abi.ai.agents.Agent.init(allocator, .{
        .name = "assistant",
        .enable_history = true,
        .temperature = 0.7,
        .top_p = 0.9,
    });
    defer agent.deinit();

    // Use the agent
    const response = try agent.chat("Hello, how can you help me?", allocator);
    defer allocator.free(response);

    std.debug.print("Agent: {s}\n", .{response});
}
```

### Agent with File-Based Configuration

When loading agent configurations from files, use the I/O backend:

```zig
const std = @import("std");
const abi = @import("abi");

pub fn loadAgentConfig(allocator: std.mem.Allocator, config_path: []const u8) !abi.ai.agents.AgentConfig {
    // Initialize I/O backend for file reading
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Read configuration file
    const content = std.Io.Dir.cwd().readFileAlloc(
        io,
        config_path,
        allocator,
        .limited(1 * 1024 * 1024),  // 1MB limit
    ) catch |err| {
        std.log.err("Failed to read config: {t}", .{err});
        return err;
    };
    defer allocator.free(content);

    // Parse JSON configuration (example)
    // Return parsed config...
    return .{
        .name = "configured-agent",
        .enable_history = true,
        .temperature = 0.7,
        .top_p = 0.9,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = try loadAgentConfig(allocator, "agent-config.json");
    var agent = try abi.ai.agents.Agent.init(allocator, config);
    defer agent.deinit();
}
```

### Agent with Environment Variables

For agents that need environment variable access (API keys, etc.):

```zig
const std = @import("std");
const abi = @import("abi");

pub fn createAgentWithEnv(allocator: std.mem.Allocator) !*abi.ai.agents.Agent {
    // Use full environment access for API key retrieval
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.init(),  // Full env access
    });
    defer io_backend.deinit();
    const io = io_backend.io();
    _ = io;  // Used for env-based operations

    // Environment variables are typically accessed via std.process.getEnvVarOwned
    // or through connector configuration

    var agent = try abi.ai.agents.Agent.init(allocator, .{
        .name = "api-agent",
        .enable_history = true,
        .temperature = 0.7,
    });

    return agent;
}
```

## Agent Configuration

### AgentConfig Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `[]const u8` | Required | Agent identifier |
| `enable_history` | `bool` | `true` | Enable conversation history |
| `temperature` | `f32` | `0.7` | Sampling temperature (0.0-2.0) |
| `top_p` | `f32` | `0.9` | Nucleus sampling parameter (0.0-1.0) |

### Agent Methods

```zig
// Core conversation methods
const response = try agent.chat(input, allocator);      // Conversational interface
const response = try agent.process(input, allocator);   // Alternative naming

// History management
const count = agent.historyCount();           // Get history entry count
const history = agent.historySlice();         // Get conversation history
agent.clearHistory();                         // Clear history

// Parameter adjustment
agent.setTemperature(0.5);                    // Update temperature
agent.setTopP(0.85);                          // Update top_p
agent.setHistoryEnabled(false);               // Disable history
```

## Backend Configuration

Agents support multiple backends through connectors:

### Echo Backend (Testing)

```zig
var agent = try abi.ai.agents.Agent.init(allocator, .{
    .name = "test-agent",
    .backend = .echo,  // Returns input as output
});
```

### OpenAI Backend

Requires `ABI_OPENAI_API_KEY` environment variable:

```zig
var agent = try abi.ai.agents.Agent.init(allocator, .{
    .name = "openai-agent",
    .backend = .openai,
    .model = "gpt-4",
    .system_prompt = "You are a helpful assistant.",
});
```

### Ollama Backend (Local)

Requires Ollama server running locally:

```zig
var agent = try abi.ai.agents.Agent.init(allocator, .{
    .name = "local-agent",
    .backend = .ollama,
    .model = "llama2",  // Or ABI_OLLAMA_MODEL env var
});
```

### HuggingFace Backend

Requires `ABI_HF_API_TOKEN` environment variable:

```zig
var agent = try abi.ai.agents.Agent.init(allocator, .{
    .name = "hf-agent",
    .backend = .huggingface,
    .model = "meta-llama/Llama-2-7b-chat-hf",
});
```

## Error Handling

### Error Context

```zig
const agent = abi.ai.agents;

// Create detailed error context
const ctx = agent.ErrorContext.apiError(
    err,
    .openai,
    "/v1/chat/completions",
    500,
    "gpt-4",
);
ctx.log();  // Logs: "AgentError: HttpRequestFailed during API request [backend=openai] [model=gpt-4]"
```

### Error Types

| Error | Description |
|-------|-------------|
| `Timeout` | Request timed out |
| `ConnectionRefused` | Cannot connect to backend |
| `ModelNotFound` | Specified model not available |
| `HttpRequestFailed` | HTTP request failed |
| `InvalidResponse` | Response parsing failed |

## CLI Usage

```bash
# Interactive agent session
zig build run -- agent

# Single message
zig build run -- agent --message "What is Zig?"

# With persona
zig build run -- agent --persona coding-assistant

# Show agent info
zig build run -- agent --info
```

## Best Practices

1. **Initialize I/O backend for file operations**: When loading configs or models from disk, create the `std.Io.Threaded` backend.

2. **Use `Environ.empty` for library code**: Only use full environment access in CLI applications.

3. **Scope I/O backend narrowly**: Create and defer cleanup immediately.

4. **Handle errors with context**: Use `ErrorContext` factory methods for detailed error reporting.

5. **Clean up resources**: Always `defer agent.deinit()` after initialization.

## Related Documentation

- [AI Module Guide](ai.md) - Full AI module documentation
- [Explore Guide](explore.md) - Codebase exploration with AI
- [Framework Guide](framework.md) - Configuration and initialization
- [Zig 0.16 Migration](migration/zig-0.16-migration.md) - Full migration guide

---

<p align="center">
  <a href="docs-index.md">&larr; Documentation Index</a> &bull;
  <a href="ai.md">AI Guide &rarr;</a>
</p>
