---
layout: documentation
title: "API Reference"
description: "Complete API reference for ABI with detailed function documentation"
---

# ABI API Reference

## 🗄️ Database Feature (`abi.features.database`)

### `database.database.Db`
Primary interface for working with the WDBX vector store.

#### Constructors
- `open(path: []const u8, create_if_missing: bool) DbError!*Db` — open or create
a database file.

#### Core Methods
- `init(self: *Db, dim: u16) DbError!void` — initialise file metadata for the
  given dimensionality.
- `addEmbedding(self: *Db, embedding: []const f32) DbError!u64` — append a
  vector and return its identifier.
- `search(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator)
  DbError![]Result` — perform ANN search (caller frees the returned slice).
- `close(self: *Db) void` — flush data and close the file handle.

#### Common Errors
- `DbError.DimensionMismatch` — vector length differs from the configured
  dimensionality.
- `DbError.CorruptedDatabase` — header or on-disk layout is invalid.
- `DbError.OutOfMemory` — allocation failed while performing the operation.

## 🧠 AI Feature (`abi.features.ai`)

### `ai.agent.Agent`
Lightweight persona-driven agent with bounded history.

- `init(allocator: std.mem.Allocator, config: AgentConfig) AgentError!*Agent`
- `process(self: *Agent, input: []const u8, allocator: std.mem.Allocator)
  AgentError![]const u8`
- `clearHistory(self: *Agent) void`
- `deinit(self: *Agent) void`

Key config fields: `name`, `persona`, `enable_history`, `max_history_items`.

### `ai.neural.NeuralNetwork`
Composable neural network pipeline for experimentation.

- `init(allocator: std.mem.Allocator, config: TrainingConfig) !*NeuralNetwork`
- `addLayer(self: *NeuralNetwork, config: LayerConfig) !void`
- `forward(self: *NeuralNetwork, input: []const f32) ![]f32`
- `deinit(self: *NeuralNetwork) void`

Supporting types:
- `LayerConfig` — layer type (`.Dense`, `.Embedding`, etc.), sizes, activation.
- `TrainingConfig` — learning rate, batch size, precision, checkpoint options.

## ⚡ SIMD & Shared Utilities

### `abi.simd`
- `add(result: []f32, a: []const f32, b: []const f32) void`
- `subtract(result: []f32, a: []const f32, b: []const f32) void`
- `multiply(result: []f32, a: []const f32, b: []const f32) void`
- `normalize(result: []f32, input: []const f32) void`

### `abi.utils`
Collection of helpers grouped under `shared/utils/*` (JSON, HTTP, math, crypto,
filesystem). Refer to the module organisation guide for detailed namespaces.

## 🔌 Framework & Plugin Runtime (`abi.framework`)

### `framework.Framework`
Coordinates feature toggles and plugin lifecycle.

- `init(allocator: std.mem.Allocator, options: FrameworkOptions) !Framework`
  (via `abi.init`).
- `refreshPlugins(self: *Framework) !void` — discover plugins across configured
  paths.
- `addPluginPath(self: *Framework, path: []const u8) !void`
- `pluginRegistry(self: *Framework) *abi.shared.registry.PluginRegistry` — access
  registry APIs from `src/shared/registry.zig`.
- `enableFeature(self: *Framework, feature: abi.framework.config.Feature) bool`
- `disableFeature(self: *Framework, feature: abi.framework.config.Feature) bool`
- `writeSummary(self: *const Framework, writer: anytype) !void`

### `FrameworkOptions`
Configure runtime behaviour.

```zig
const options = abi.framework.FrameworkOptions{
    .auto_discover_plugins = true,
    .auto_register_plugins = true,
    .plugin_paths = &.{ "./plugins" },
};
```

Feature toggles are derived via `abi.framework.deriveFeatureToggles(options)` and
mirror the `FeatureTag` enum in `src/features/mod.zig`.

### Plugin Registry (`abi.shared`)

`framework.pluginRegistry()` exposes the shared registry with methods such as:
- `loadPlugin(path: []const u8) PluginError!void`
- `startAllPlugins(self: *PluginRegistry) !void`
- `getPluginCount(self: *PluginRegistry) usize`
- `writeStatus(self: *PluginRegistry, writer: anytype) !void`

Plugin metadata types live under `abi.shared.types` (`PluginInfo`, `PluginType`,
`PluginConfig`).

## 🌐 Web & Connectors (`abi.features.web`)

Key modules include `http_client.zig` and `web_server.zig`.

- `HttpClient.init(allocator: std.mem.Allocator, config: HttpClient.Config)
  !*HttpClient`
- `HttpClient.get(self: *HttpClient, url: []const u8, allocator: std.mem.Allocator)
  ![]u8`
- `HttpClient.deinit(self: *HttpClient) void`

Connector modules follow the same pattern but wrap external APIs or plugin
bridges.

## 🧾 Notes

- All slices returned from feature modules (database search results, neural
  network outputs, HTTP responses) must be freed by the caller using the
  allocator passed into the call.
- Namespaces are stabilised behind `abi.features.*`, so prefer those imports over
  historical `abi.database`/`abi.ai` references.
