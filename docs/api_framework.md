# framework API Reference

> Framework orchestration and lifecycle management

**Source:** [`src/framework.zig`](../../src/framework.zig)

---

Framework Orchestration Layer

Manages the lifecycle of the ABI framework, coordinating feature
initialization, configuration, and runtime state.

## Usage

```zig
const abi = @import("abi");

// Using init with defaults
var fw = try abi.init(allocator);
defer fw.deinit();

// Using builder pattern
var fw = try abi.Framework.builder(allocator)
.withGpu(.{ .backend = .vulkan })
.withAi(.{ .llm = .{} })
.build();
defer fw.deinit();

// Check feature status
if (fw.isEnabled(.gpu)) {
// Use GPU features
}
```

---

## API

### `pub const Framework`

<sup>**type**</sup>

Framework orchestration handle.
Manages lifecycle of all enabled features.

### `pub fn init(allocator: std.mem.Allocator, cfg: Config) Error!Framework`

<sup>**fn**</sup>

Initialize the framework with the given configuration.

### `pub fn initWithIo(allocator: std.mem.Allocator, cfg: Config, io: std.Io) Error!Framework`

<sup>**fn**</sup>

Initialize the framework with the given configuration **and** an I/O backend.
This method is used by the builder when `withIo` is supplied.

### `pub fn initDefault(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup>

Create a framework with default configuration.

### `pub fn initMinimal(allocator: std.mem.Allocator) Error!Framework`

<sup>**fn**</sup>

Create a framework with minimal configuration (no features enabled).

### `pub fn builder(allocator: std.mem.Allocator) FrameworkBuilder`

<sup>**fn**</sup>

Start building a framework configuration.

### `pub fn deinit(self: *Framework) void`

<sup>**fn**</sup>

Shutdown and cleanup the framework.

### `pub fn isRunning(self: *const Framework) bool`

<sup>**fn**</sup>

Check if the framework is running.

### `pub fn isEnabled(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is enabled.

### `pub fn getState(self: *const Framework) State`

<sup>**fn**</sup>

Get the current framework state.

### `pub fn getGpu(self: *Framework) Error!*gpu_mod.Context`

<sup>**fn**</sup>

Get GPU context (returns error if not enabled).

### `pub fn getAi(self: *Framework) Error!*ai_mod.Context`

<sup>**fn**</sup>

Get AI context (returns error if not enabled).

### `pub fn getDatabase(self: *Framework) Error!*database_mod.Context`

<sup>**fn**</sup>

Get database context (returns error if not enabled).

### `pub fn getNetwork(self: *Framework) Error!*network_mod.Context`

<sup>**fn**</sup>

Get network context (returns error if not enabled).

### `pub fn getObservability(self: *Framework) Error!*observability_mod.Context`

<sup>**fn**</sup>

Get observability context (returns error if not enabled).

### `pub fn getWeb(self: *Framework) Error!*web_mod.Context`

<sup>**fn**</sup>

Get web context (returns error if not enabled).

### `pub fn getRuntime(self: *Framework) *runtime_mod.Context`

<sup>**fn**</sup>

Get runtime context (always available).

### `pub fn getRegistry(self: *Framework) *Registry`

<sup>**fn**</sup>

Get the feature registry for runtime feature management.

### `pub fn isFeatureRegistered(self: *const Framework, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is registered in the registry.

### `pub fn listRegisteredFeatures(self: *const Framework, allocator: std.mem.Allocator) Registry.Error![]Feature`

<sup>**fn**</sup>

List all registered features.

### `pub const FrameworkBuilder`

<sup>**type**</sup>

Fluent builder for Framework initialization.

### `pub fn withDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Start with default configuration.

### `pub fn withGpu(self: *FrameworkBuilder, gpu_config: config_module.GpuConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable GPU with configuration.

### `pub fn withGpuDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable GPU with defaults.

### `pub fn withIo(self: *FrameworkBuilder, io: std.Io) *FrameworkBuilder`

<sup>**fn**</sup>

Provide a shared I/O backend for the framework.
Pass the `std.Io` obtained from `IoBackend.init`.

### `pub fn withAi(self: *FrameworkBuilder, ai_config: config_module.AiConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable AI with configuration.

### `pub fn withAiDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable AI with defaults.

### `pub fn withLlm(self: *FrameworkBuilder, llm_config: config_module.LlmConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable LLM only.

### `pub fn withDatabase(self: *FrameworkBuilder, db_config: config_module.DatabaseConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable database with configuration.

### `pub fn withDatabaseDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable database with defaults.

### `pub fn withNetwork(self: *FrameworkBuilder, net_config: config_module.NetworkConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable network with configuration.

### `pub fn withNetworkDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable network with defaults.

### `pub fn withObservability(self: *FrameworkBuilder, obs_config: config_module.ObservabilityConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable observability with configuration.

### `pub fn withObservabilityDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable observability with defaults.

### `pub fn withWeb(self: *FrameworkBuilder, web_config: config_module.WebConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Enable web with configuration.

### `pub fn withWebDefaults(self: *FrameworkBuilder) *FrameworkBuilder`

<sup>**fn**</sup>

Enable web with defaults.

### `pub fn withPlugins(self: *FrameworkBuilder, plugin_config: config_module.PluginConfig) *FrameworkBuilder`

<sup>**fn**</sup>

Configure plugins.

### `pub fn build(self: *FrameworkBuilder) Framework.Error!Framework`

<sup>**fn**</sup>

Build and initialize the framework.
If an I/O backend was supplied via `withIo`, it will be stored in the
resulting `Framework` instance.

### `pub const FrameworkOptions`

<sup>**type**</sup>

Legacy FrameworkOptions for backward compatibility.
@deprecated Use Config directly.

### `pub fn toConfig(self: FrameworkOptions) Config`

<sup>**fn**</sup>

Convert to new Config format.

### `pub const FrameworkConfiguration`

<sup>**const**</sup>

Legacy FrameworkConfiguration for backward compatibility.
@deprecated Use Config directly.

### `pub const RuntimeConfig`

<sup>**const**</sup>

Legacy RuntimeConfig for backward compatibility.
@deprecated Use Config directly.

### `pub fn runtimeConfigFromOptions(allocator: std.mem.Allocator, options: FrameworkOptions) !Config`

<sup>**fn**</sup>

Convert legacy options to new config.
@deprecated Use Config directly.

---

*Generated automatically by `zig build gendocs`*
