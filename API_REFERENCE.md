# API Reference (Concise)

> For detailed usage guides, see the [Documentation Index](docs/intro.md).

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

**Security Note for backup/restore**:

- Backup and restore operations are restricted to the `backups/` directory only
- Filenames must not contain path traversal sequences (`..`), absolute paths, or Windows drive letters
- Invalid filenames will return `PathValidationError`
- The `backups/` directory is created automatically if it doesn't exist
- This restriction prevents path traversal attacks (see SECURITY.md for details)

## Compute Engine API

- `abi.compute.runtime.Engine` - Main compute runtime
- `abi.compute.runtime.runWorkload(engine, workload, timeout_ms)` -> `!Result`
- `abi.compute.runtime.registerWorkloadType(name, vtable)`

**Timeout Semantics**:

- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if result not ready
- `timeout_ms>0`: Waits for the specified timeout (in milliseconds) before returning `EngineError.Timeout`
- `timeout_ms=null`: Waits indefinitely until result is ready

**Breaking Change (0.2.1)**: Prior to version 0.2.1, `timeout_ms=0` returned `ResultNotFound` after one check. This behavior has changed to return `EngineError.Timeout` immediately for clarity. Migration: Use `timeout_ms=1000` for a one-second timeout.

## Connectors API

External service integrations accessible via `abi.connectors`:

- `abi.connectors.openai` - OpenAI GPT-4, GPT-3.5, embeddings
- `abi.connectors.ollama` - Local LLM inference via Ollama
- `abi.connectors.huggingface` - HuggingFace Inference API
- `abi.connectors.discord` - Discord Bot API (REST, webhooks, interactions)

**Discord Convenience Exports:**

- `abi.discord` - Direct access to Discord connector
- `abi.DiscordClient` - Discord REST API client
- `abi.DiscordConfig` - Configuration struct
- `abi.DiscordTools` - AI agent tools for Discord

## Modules

- `src/core` - I/O, diagnostics, collections
- `src/features` - feature modules (AI, GPU, database, web, monitoring, connectors)
- `src/framework` - orchestration runtime and lifecycle management
- `src/shared` - shared utilities and platform helpers
- `src/compute` - compute runtime, memory management, concurrency

## Contacts

src/shared/contacts.zig provides a centralized list of maintainer contacts extracted from the repository markdown files. Import this module wherever contact information is needed.

