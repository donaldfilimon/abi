# ai API Reference

> AI module with agents, LLM, embeddings, and training

**Source:** [`src/ai/mod.zig`](../../src/ai/mod.zig)

---

AI Module - Public API

This is the primary entry point for AI functionality. Import from here for
Framework integration and the stable public API.

Modular AI capabilities organized as independent sub-features:

- **core**: Shared types, interfaces, and utilities (always available when AI enabled)
- **llm**: Local LLM inference (GGUF, transformer models)
- **embeddings**: Vector embeddings generation
- **agents**: AI agent runtime and tools
- **training**: Model training pipelines

Each sub-feature can be independently enabled/disabled.

## Usage

```zig
const ai = @import("ai/mod.zig");

// Initialize AI context
var ctx = try ai.Context.init(allocator, .{
.llm = .{ .model_path = "./models/llama.gguf" },
});
defer ctx.deinit();

// Use LLM
const response = try ctx.getLlm().generate("Hello, world!");
```

---

## API

### `pub const core`

<sup>**type**</sup>

Core AI types and utilities (always available when AI enabled)

### `pub const llm`

<sup>**const**</sup>

LLM inference module

### `pub const embeddings`

<sup>**const**</sup>

Embeddings generation module

### `pub const agents`

<sup>**const**</sup>

Agent runtime module

### `pub const training`

<sup>**const**</sup>

Training pipelines module

### `pub const vision`

<sup>**const**</sup>

Vision/image processing module

### `pub const Context`

<sup>**type**</sup>

AI context for Framework integration.
Manages AI sub-features based on configuration.

### `pub fn getLlm(self: *Context) Error!*llm.Context`

<sup>**fn**</sup>

Get LLM context (returns error if not enabled).

### `pub fn getEmbeddings(self: *Context) Error!*embeddings.Context`

<sup>**fn**</sup>

Get embeddings context (returns error if not enabled).

### `pub fn getAgents(self: *Context) Error!*agents.Context`

<sup>**fn**</sup>

Get agents context (returns error if not enabled).

### `pub fn getTraining(self: *Context) Error!*training.Context`

<sup>**fn**</sup>

Get training context (returns error if not enabled).

### `pub fn isSubFeatureEnabled(self: *Context, feature: SubFeature) bool`

<sup>**fn**</sup>

Check if a sub-feature is enabled.

### `pub fn isEnabled() bool`

<sup>**fn**</sup>

Check if AI is enabled at compile time.

### `pub fn isLlmEnabled() bool`

<sup>**fn**</sup>

Check if LLM is enabled at compile time.

### `pub fn isInitialized() bool`

<sup>**fn**</sup>

Check if AI module is initialized.

### `pub fn init(allocator: std.mem.Allocator) Error!void`

<sup>**fn**</sup>

Initialize the AI module (legacy compatibility).

### `pub fn deinit() void`

<sup>**fn**</sup>

Deinitialize the AI module (legacy compatibility).

---

*Generated automatically by `zig build gendocs`*
