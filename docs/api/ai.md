# ai API Reference

> AI module with agents, LLM, embeddings, and training

**Source:** [`src/ai/mod.zig`](../../src/ai/mod.zig)

---

AI Module - Public API

This is the primary entry point for AI functionality in the ABI framework.
Import from here for Framework integration and the stable public API.

## Overview

The AI module provides modular AI capabilities organized as independent sub-features,
each of which can be enabled or disabled independently:

| Sub-feature | Description | Build Flag |
|-------------|-------------|------------|
| **core** | Shared types, interfaces, utilities | Always available |
| **llm** | Local LLM inference (GGUF models) | `-Denable-llm` |
| **embeddings** | Vector embeddings generation | `-Denable-ai` |
| **agents** | AI agent runtime and tools | `-Denable-ai` |
| **training** | Model training pipelines | `-Denable-ai` |
| **personas** | Multi-persona AI assistant | `-Denable-ai` |
| **vision** | Image processing and analysis | `-Denable-vision` |

## Quick Start

```zig
const abi = @import("abi");

// Initialize framework with AI enabled
var fw = try abi.Framework.init(allocator, .{
.ai = .{
.llm = .{ .model_path = "./models/llama-7b.gguf" },
.embeddings = .{ .dimension = 768 },
},
});
defer fw.deinit();

// Access AI context
const ai_ctx = try fw.getAi();

// Use LLM
const llm = try ai_ctx.getLlm();
// ... perform inference ...
```

## Standalone Usage

```zig
const ai = abi.ai;

// Initialize AI context directly
var ctx = try ai.Context.init(allocator, .{
.llm = .{ .model_path = "./models/llama.gguf" },
});
defer ctx.deinit();

// Check which sub-features are enabled
if (ctx.isSubFeatureEnabled(.llm)) {
const llm = try ctx.getLlm();
// ... use LLM ...
}
```

## Sub-module Access

Access sub-modules directly through the namespace:
- `abi.ai.llm` - LLM inference engine
- `abi.ai.embeddings` - Embedding generation
- `abi.ai.agents` - Agent runtime
- `abi.ai.training` - Training pipelines
- `abi.ai.personas` - Multi-persona system
- `abi.ai.vision` - Vision processing

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

### `pub const database`

<sup>**const**</sup>

AI Database module

### `pub const vision`

<sup>**const**</sup>

Vision/image processing module

### `pub const documents`

<sup>**const**</sup>

Document understanding and processing module

### `pub const Context`

<sup>**type**</sup>

AI context for Framework integration.

The Context struct manages all AI sub-features (LLM, embeddings, agents, training,
personas) based on the provided configuration. Each sub-feature is independently
initialized and can be accessed through type-safe getter methods.

## Thread Safety

The Context itself is not thread-safe. If you need to access AI features from
multiple threads, use external synchronization.

## Memory Management

The Context allocates memory for each enabled sub-feature context. All memory
is released when `deinit()` is called.

## Example

```zig
var ctx = try ai.Context.init(allocator, .{
.llm = .{ .model_path = "./models/llama.gguf" },
.embeddings = .{ .dimension = 768 },
});
defer ctx.deinit();

// Access sub-features
const llm = try ctx.getLlm();
const emb = try ctx.getEmbeddings();
```

### `pub fn init(allocator: std.mem.Allocator, cfg: config_module.AiConfig) !*Context`

<sup>**fn**</sup>

Initialize the AI context with the given configuration.

## Parameters

- `allocator`: Memory allocator for context resources
- `cfg`: AI configuration specifying which sub-features to enable

## Returns

A pointer to the initialized Context.

## Errors

- `error.AiDisabled`: AI is disabled at compile time
- `error.OutOfMemory`: Memory allocation failed
- Sub-feature specific errors during initialization

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

### `pub fn getPersonas(self: *Context) Error!*personas.Context`

<sup>**fn**</sup>

Get personas context (returns error if not enabled).

### `pub fn isSubFeatureEnabled(self: *Context, feature: SubFeature) bool`

<sup>**fn**</sup>

Check if a sub-feature is enabled.

### `pub fn getDiscoveredModels(self: *Context) []discovery.DiscoveredModel`

<sup>**fn**</sup>

Get discovered models (returns empty slice if discovery not enabled).

### `pub fn discoveredModelCount(self: *Context) usize`

<sup>**fn**</sup>

Get number of discovered models.

### `pub fn findBestModel(self: *Context, requirements: discovery.ModelRequirements) ?*discovery.DiscoveredModel`

<sup>**fn**</sup>

Find best model matching requirements.

### `pub fn generateAdaptiveConfig(self: *Context, model: *const discovery.DiscoveredModel) discovery.AdaptiveConfig`

<sup>**fn**</sup>

Generate adaptive configuration for a model.

### `pub fn getCapabilities(self: *const Context) discovery.SystemCapabilities`

<sup>**fn**</sup>

Get system capabilities.

### `pub fn addModelPath(self: *Context, path: []const u8) !void`

<sup>**fn**</sup>

Add a model path to the discovery system.
Use this to register known model files.

### `pub fn addModelWithSize(self: *Context, path: []const u8, size_bytes: u64) !void`

<sup>**fn**</sup>

Add a model path with known file size.

### `pub fn clearDiscoveredModels(self: *Context) void`

<sup>**fn**</sup>

Clear all discovered models.

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
