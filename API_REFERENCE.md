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

- `abi.compute.DistributedComputeEngine` - Main compute runtime (alias: `abi.compute.runtime.DistributedComputeEngine`)
- `abi.compute.createDefaultEngine(allocator)` -> `!Engine` - Create engine with default config
- `abi.compute.createEngine(allocator, config)` -> `!Engine` - Create engine with custom config
- `abi.compute.submitTask(engine, ResultType, task)` -> `!TaskId` - Submit a task for execution
- `abi.compute.waitForResult(engine, ResultType, id, timeout_ms)` -> `!Result` - Wait for task result
- `abi.compute.runTask(engine, ResultType, task, timeout_ms)` -> `!Result` - Submit and wait for result
- `abi.compute.runWorkload(engine, ResultType, workload, timeout_ms)` -> `!Result` - Alias for runTask

**Timeout Semantics**:

- `timeout_ms=0`: Immediately returns `EngineError.Timeout` if result not ready
- `timeout_ms>0`: Waits for the specified timeout (in milliseconds) before returning `EngineError.Timeout`
- `timeout_ms=null`: Waits indefinitely until result is ready

**Breaking Change (0.2.1)**: Prior to version 0.2.1, `timeout_ms=0` returned `ResultNotFound` after one check. This behavior has changed to return `EngineError.Timeout` immediately for clarity. Migration: Use `timeout_ms=1000` for a one-second timeout.

## AI & Agent API

- `abi.ai.Agent` - Conversational agent with history and configuration
- `abi.ai.Agent.init(allocator, config)` -> `!Agent` - Create a new agent
- `abi.ai.Agent.deinit()` - Clean up agent resources
- `abi.ai.Agent.process(input, allocator)` -> `![]u8` - Process input and return response
- `abi.ai.Agent.chat(input, allocator)` -> `![]u8` - Alias for process() providing chat interface
- `abi.ai.train(allocator, config)` -> `!TrainingReport` - Run training pipeline
- `abi.ai.federated.Coordinator` - Federated learning coordinator
- `abi.ai.federated.Coordinator.init(allocator, config, model_size)` -> `!Coordinator`
- `abi.ai.federated.Coordinator.registerNode(node_id)` -> `!void`
- `abi.ai.federated.Coordinator.submitUpdate(update)` -> `!void`
- `abi.ai.federated.Coordinator.aggregate()` -> `![]f32` - Aggregate updates

## Connectors API

- `abi.connectors.openai` - OpenAI API connector
- `abi.connectors.ollama` - Ollama API connector
- `abi.connectors.huggingface` - HuggingFace API connector

Each connector provides:
- `Client.init(allocator, config)` - Initialize client
- `Client.deinit()` - Clean up resources
- Connector-specific methods for inference/chat/completion

## Modules

- `lib/core` - I/O, diagnostics, collections
- `lib/features` - feature modules
- `lib/framework` - orchestration runtime
- `lib/shared` - shared utilities and platform helpers
