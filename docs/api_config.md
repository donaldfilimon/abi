# config API Reference

> Unified configuration system with builder pattern

**Source:** [`src/config.zig`](../../src/config.zig)

---

Unified Configuration System

Single source of truth for all ABI framework configuration.
Supports both struct literal and builder pattern APIs.

## Usage

```zig
// Minimal - everything auto-detected with defaults
var fw = try abi.init(allocator);

// Struct literal style
var fw = try abi.init(allocator, .{
.gpu = .{ .backend = .vulkan },
.ai = .{ .llm = .{} },
});

// Builder style
var fw = try abi.Framework.builder(allocator)
.withGpu(.{ .backend = .cuda })
.withAi(.{ .llm = .{ .model_path = "./models/llama.gguf" } })
.withDatabase(.{ .path = "./data" })
.build();
```

---

## API

### `pub const Config`

<sup>**type**</sup>

Unified configuration for the ABI framework.
Each field being non-null enables that feature with the specified settings.
A null field means the feature is disabled.

### `pub fn defaults() Config`

<sup>**fn**</sup>

Returns a config with all compile-time enabled features activated with defaults.

### `pub fn minimal() Config`

<sup>**fn**</sup>

Returns a minimal config with no features enabled.

### `pub fn isEnabled(self: Config, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is enabled in this config.

### `pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature`

<sup>**fn**</sup>

Get list of enabled features.

### `pub const Feature`

<sup>**type**</sup>

Feature identifiers for the framework.

### `pub fn isCompileTimeEnabled(self: Feature) bool`

<sup>**fn**</sup>

Check if this feature is available at compile time.

### `pub fn autoSelectBackend() Backend`

<sup>**fn**</sup>

Select the best backend based on availability.

### `pub fn llmOnly(config: LlmConfig) AiConfig`

<sup>**fn**</sup>

Enable only LLM inference.

### `pub fn embeddingsOnly(config: EmbeddingsConfig) AiConfig`

<sup>**fn**</sup>

Enable only embeddings.

### `pub fn inMemory() DatabaseConfig`

<sup>**fn**</sup>

In-memory database configuration.

### `pub fn standalone() NetworkConfig`

<sup>**fn**</sup>

Standalone node (no clustering).

### `pub fn distributed() NetworkConfig`

<sup>**fn**</sup>

Distributed compute with unified memory.

### `pub fn thunderbolt() UnifiedMemoryConfig`

<sup>**fn**</sup>

High-performance for local Thunderbolt links.

### `pub fn internet() UnifiedMemoryConfig`

<sup>**fn**</sup>

Secure for Internet links.

### `pub fn highPerformance() LinkingConfig`

<sup>**fn**</sup>

High-performance local linking.

### `pub fn secure() LinkingConfig`

<sup>**fn**</sup>

Secure Internet linking.

### `pub fn full() ObservabilityConfig`

<sup>**fn**</sup>

Full observability (all features enabled).

### `pub const Builder`

<sup>**type**</sup>

Fluent builder for constructing Config instances.

### `pub fn withDefaults(self: *Builder) *Builder`

<sup>**fn**</sup>

Start with default configuration for all compile-time enabled features.

### `pub fn withGpu(self: *Builder, gpu_config: GpuConfig) *Builder`

<sup>**fn**</sup>

Enable GPU with specified configuration.

### `pub fn withGpuDefaults(self: *Builder) *Builder`

<sup>**fn**</sup>

Enable GPU with default configuration.

### `pub fn withAi(self: *Builder, ai_config: AiConfig) *Builder`

<sup>**fn**</sup>

Enable AI with specified configuration.

### `pub fn withAiDefaults(self: *Builder) *Builder`

<sup>**fn**</sup>

Enable AI with default configuration.

### `pub fn withLlm(self: *Builder, llm_config: LlmConfig) *Builder`

<sup>**fn**</sup>

Enable LLM only (convenience method).

### `pub fn withDatabase(self: *Builder, db_config: DatabaseConfig) *Builder`

<sup>**fn**</sup>

Enable database with specified configuration.

### `pub fn withDatabaseDefaults(self: *Builder) *Builder`

<sup>**fn**</sup>

Enable database with default configuration.

### `pub fn withNetwork(self: *Builder, net_config: NetworkConfig) *Builder`

<sup>**fn**</sup>

Enable network with specified configuration.

### `pub fn withNetworkDefaults(self: *Builder) *Builder`

<sup>**fn**</sup>

Enable network with default configuration.

### `pub fn withObservability(self: *Builder, obs_config: ObservabilityConfig) *Builder`

<sup>**fn**</sup>

Enable observability with specified configuration.

### `pub fn withObservabilityDefaults(self: *Builder) *Builder`

<sup>**fn**</sup>

Enable observability with default configuration.

### `pub fn withWeb(self: *Builder, web_config: WebConfig) *Builder`

<sup>**fn**</sup>

Enable web with specified configuration.

### `pub fn withWebDefaults(self: *Builder) *Builder`

<sup>**fn**</sup>

Enable web with default configuration.

### `pub fn withPlugins(self: *Builder, plugin_config: PluginConfig) *Builder`

<sup>**fn**</sup>

Configure plugins.

### `pub fn build(self: *Builder) Config`

<sup>**fn**</sup>

Finalize and return the configuration.

### `pub fn validate(config: Config) ConfigError!void`

<sup>**fn**</sup>

Validate configuration against compile-time constraints.

---

*Generated automatically by `zig build gendocs`*
