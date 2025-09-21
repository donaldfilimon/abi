---
layout: documentation
title: "Module Reference"
description: "Comprehensive reference for all ABI modules and components"
---

# ABI Module Reference

## 🚩 Entrypoint

### `abi` — Main Module
Central import that wires the framework runtime, feature namespaces, and shared
libraries. The file `src/mod.zig` exposes curated accessors instead of direct
path imports so consumers can stay aligned with internal refactors.

#### Key Exports
- `abi.features` — grouped feature families (`ai`, `database`, `gpu`, `web`,
  `monitoring`, `connectors`)
- `abi.framework` — runtime orchestration, feature toggles, and lifecycle APIs
- `abi.utils` / `abi.core` / `abi.logging` / `abi.platform` — shared utility
  layers
- `abi.simd` — SIMD helpers re-exported from `src/shared/simd.zig`

## 🧩 Feature Namespaces (`abi.features.*`)

### `abi.features.ai`
Comprehensive AI toolkit including agents, enhanced transformer pipelines,
distributed training, and model registries. Major files:

- `agent.zig` / `enhanced_agent.zig` — persona-driven assistants with history
  management
- `transformer.zig`, `reinforcement_learning.zig`, `neural.zig` — model
  implementations
- `model_registry.zig`, `model_serialization.zig` — lifecycle management for AI
  assets

Usage:

```zig
const ai = abi.features.ai;
var agent = try ai.agent.Agent.init(allocator, .{ .name = "Helper" });
defer agent.deinit();
```

### `abi.features.database`
WDBX vector database engine and surrounding tooling.

- `database.zig` — storage engine with SIMD-accelerated search
- `config.zig` — `WdbxConfig` parsing/validation
- `http.zig` / `cli.zig` — service surfaces layered on the core engine
- `database_sharding.zig` / `unified.zig` — distributed operations

Usage:

```zig
const database = abi.features.database;
var db = try database.database.Db.open("vectors.wdbx", true);
defer db.close();
try db.init(768);
```

### `abi.features.gpu`
GPU accelerators, memory orchestration, backend selection, and benchmarking.
Subdirectories include `compute/`, `memory/`, `backends/`, `libraries/`, and
`testing/` for platform-specific functionality.

### `abi.features.web`
HTTP clients/servers, C bindings, and demos that surface ABI functionality over
the network. Core files: `http_client.zig`, `web_server.zig`, `wdbx_http.zig`,
and `weather.zig`.

### `abi.features.monitoring`
Telemetry, profiling, regression tooling, and metrics exporters consolidated in
`monitoring/mod.zig` and supporting files.

### `abi.features.connectors`
Bridges to third-party APIs and plugin wrappers (e.g. `plugin.zig`) that let the
runtime load external capability providers.

## 🧠 Framework Runtime (`abi.framework.*`)

The framework namespace coordinates features, plugin discovery, and lifecycle
management.

- `config.zig` — `FrameworkOptions`, feature toggles, labels, and descriptions
- `runtime.zig` — `Framework` struct with methods such as `refreshPlugins`,
  `enableFeature`, and `writeSummary`
- `feature_manager.zig`, `catalog.zig`, `state.zig` — advanced orchestration and
  registry plumbing

Initialization example:

```zig
var framework = try abi.init(std.heap.page_allocator, .{
    .auto_discover_plugins = true,
    .plugin_paths = &.{ "./plugins" },
});
defer framework.deinit();

try framework.refreshPlugins();
```

## 🛠 Shared Libraries (`abi.core`, `abi.utils`, `abi.logging`, `abi.platform`)

- `abi.core` → `src/shared/core/mod.zig` (error types, lifecycle, framework
  glue)
- `abi.utils` → `src/shared/utils/mod.zig` (JSON, math, crypto, HTTP, FS, and
  string helpers)
- `abi.logging` → `src/shared/logging/mod.zig` (structured logging backends)
- `abi.platform` → `src/shared/platform/mod.zig` (OS/platform detection and
  capability reporting)
- `abi.simd` → `src/shared/simd.zig` (SIMD intrinsics surfaced to features)
- `src/shared/mod.zig` — plugin façade exposing `PluginRegistry`, loader, and
  interface types used by the framework

## 🔌 Plugin System

Plugins are coordinated by the framework but implemented in the shared layer:

- Use `abi.init` with `FrameworkOptions` to configure discovery/registration
- Call `framework.pluginRegistry()` to access the shared registry API
- Plugin metadata/types live under `abi.shared.types` (via
  `@import("shared/types.zig")` inside the runtime)

Example snippet:

```zig
const registry = framework.pluginRegistry();
try registry.loadPlugin("./plugins/example_plugin.so");
std.log.info("Plugins ready: {d}", .{registry.getPluginCount()});
```

## 🚀 Usage Patterns

### Feature-Oriented Workflow

```zig
const abi = @import("abi");

var framework = try abi.init(std.heap.page_allocator, .{});
defer framework.deinit();

const ai = abi.features.ai;
var agent = try ai.agent.Agent.init(std.heap.page_allocator, .{ .name = "Ops" });
defer agent.deinit();

const reply = try agent.process("status", std.heap.page_allocator);
defer std.heap.page_allocator.free(reply);
```

### Database & HTTP Integration

```zig
const abi = @import("abi");

const database = abi.features.database;
var db = try database.database.Db.open("vectors.wdbx", true);
defer db.close();
try db.init(512);

const web = abi.features.web;
var client = try web.http_client.HttpClient.init(std.heap.page_allocator, .{});
defer client.deinit();
```

### SIMD Helpers

```zig
var buffer = try allocator.alloc(f32, 1024);
defer allocator.free(buffer);
abi.simd.normalize(buffer, buffer);
```

## 🎯 Benefits of the Feature-Centric Layout

- Aligns documentation with the actual `src/features/*` directory structure
- Keeps shared utilities clearly separated from orchestration concerns
- Encourages consumers to adopt `abi.features.*` namespaces, reducing churn
  during refactors
- Simplifies discovery: inspect `src/features/mod.zig` for feature families,
  `src/framework/` for lifecycle management, and `src/shared/` for reusable
  building blocks
