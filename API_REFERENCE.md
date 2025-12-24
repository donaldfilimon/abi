# API Reference (Concise)

This is a high-level summary of the public ABI API surface. See the source for
implementation details.

## Core Entry Points

- `abi.init(allocator, config_or_options)` -> `Framework`
- `abi.shutdown(framework)`
- `abi.version()` -> `[]const u8`
- `abi.createDefaultFramework(allocator)` -> `Framework`
- `abi.createFramework(allocator, config_or_options)` -> `Framework`

## Framework Types

- `abi.Framework`
- `abi.FrameworkOptions`
- `abi.RuntimeConfig`
- `abi.Feature` and `abi.features.FeatureTag`

## Feature Namespaces

- `abi.ai` - agent runtime, tools, training pipelines
- `abi.database` - WDBX database and helpers
- `abi.gpu` - GPU backends and vector search helpers
- `abi.web` - HTTP helpers, web utilities
- `abi.monitoring` - logging, metrics, tracing, profiling
- `abi.connectors` - connector interfaces and implementations

## WDBX Convenience API

- `abi.wdbx.createDatabase` / `connectDatabase` / `closeDatabase`
- `abi.wdbx.insertVector` / `searchVectors` / `deleteVector`
- `abi.wdbx.updateVector` / `getVector` / `listVectors`
- `abi.wdbx.getStats` / `optimize` / `backup` / `restore`

## Modules

- `lib/core` - I/O, diagnostics, collections
- `lib/features` - feature modules
- `lib/framework` - orchestration runtime
- `lib/shared` - shared utilities and platform helpers
