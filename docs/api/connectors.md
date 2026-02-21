# connectors

> External service connectors (OpenAI, Anthropic, Ollama, etc.).

**Source:** [`src/services/connectors/mod.zig`](../../src/services/connectors/mod.zig)

**Availability:** Always enabled

---

Connector configuration loaders and auth helpers.

This module provides unified access to various AI service connectors including:

- **OpenAI**: GPT models via the Chat Completions API
- **Anthropic**: Claude models via the Messages API
- **Ollama**: Local LLM inference server
- **HuggingFace**: Hosted inference API
- **Mistral**: Mistral AI models with OpenAI-compatible API
- **Cohere**: Chat, embeddings, and reranking
- **LM Studio**: Local LLM inference with OpenAI-compatible API
- **vLLM**: High-throughput local LLM serving with OpenAI-compatible API
- **MLX**: Apple Silicon-optimized inference via mlx-lm server
- **Discord**: Bot integration for Discord

## Usage

Each connector can be loaded from environment variables:

```zig
const connectors = @import("abi").connectors;

// Load and create clients
if (try connectors.tryLoadOpenAI(allocator)) |config| {
var client = try connectors.openai.Client.init(allocator, config);
defer client.deinit();
// Use client...
}
```

## Security

All connectors securely wipe API keys from memory using `std.crypto.secureZero`
before freeing to prevent memory forensics attacks.

---

## API

### `pub fn init(_: std.mem.Allocator) !void`

<sup>**fn**</sup>

Initialize the connectors subsystem (idempotent; no-op if already initialized).

### `pub fn deinit() void`

<sup>**fn**</sup>

Tear down the connectors subsystem; safe to call multiple times.

### `pub fn isEnabled() bool`

<sup>**fn**</sup>

Returns true; connectors are always available when this module is compiled in.

### `pub fn isInitialized() bool`

<sup>**fn**</sup>

Returns true after `init()` has been called.

### `pub fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8`

<sup>**fn**</sup>

Read environment variable by name; returns owned slice or null if unset. Caller must free.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
