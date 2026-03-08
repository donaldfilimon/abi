# connectors

> Connector configuration loaders and auth helpers.

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

**Source:** [`src/services/connectors/mod.zig`](../../src/services/connectors/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-fn-init-std-mem-allocator-void"></a>`pub fn init(_: std.mem.Allocator) !void`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L60)

Initialize the connectors subsystem (idempotent; no-op if already initialized).

### <a id="pub-fn-deinit-void"></a>`pub fn deinit() void`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L65)

Tear down the connectors subsystem; safe to call multiple times.

### <a id="pub-fn-isenabled-bool"></a>`pub fn isEnabled() bool`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L70)

Returns true; connectors are always available when this module is compiled in.

### <a id="pub-fn-isinitialized-bool"></a>`pub fn isInitialized() bool`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L75)

Returns true after `init()` has been called.

### <a id="pub-fn-getenvowned-allocator-std-mem-allocator-name-const-u8-u8"></a>`pub fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8`

<sup>**fn**</sup> | [source](../../src/services/connectors/mod.zig#L85)

Read environment variable by name; returns owned slice or null if unset. Caller must free.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use the `$zig-master` Codex skill for ABI Zig validation, docs generation, and build-wiring changes.
