# Configuration Module
> **Last reviewed:** 2026-01-31

The configuration module (`src/core/config/`) provides a unified, feature-aware configuration system for the ABI framework. It manages settings for all major subsystems and supports environment variable overrides through the `ConfigLoader`.

## Overview

The module uses a **modular architecture** where each feature domain (GPU, AI, Database, etc.) maintains its own configuration file. These are re-exported through `mod.zig` for convenient access.

## Key Components

### Unified Config Struct

The `Config` struct is the central configuration object:

```zig
pub const Config = struct {
    gpu: ?GpuConfig = null,
    ai: ?AiConfig = null,
    database: ?DatabaseConfig = null,
    network: ?NetworkConfig = null,
    observability: ?ObservabilityConfig = null,
    web: ?WebConfig = null,
    plugins: PluginConfig = .{},
};
```

All feature configs are **optional** — `null` means the feature is disabled.

### Feature Configurations

Each domain has its own configuration type:

| Config Type | File | Purpose |
|------------|------|---------|
| `GpuConfig` | `gpu.zig` | GPU acceleration settings (backends, memory, optimization) |
| `AiConfig` | `ai.zig` | AI core settings (LLM, embeddings, agents, training) |
| `DatabaseConfig` | `database.zig` | Vector database (WDBX) settings |
| `NetworkConfig` | `network.zig` | Distributed compute and network settings |
| `ObservabilityConfig` | `observability.zig` | Metrics, tracing, and profiling configuration |
| `WebConfig` | `web.zig` | HTTP server and web utilities settings |
| `PluginConfig` | `plugin.zig` | Plugin system settings |

### Feature Enum

The `Feature` enum provides compile-time checking and introspection:

```zig
pub const Feature = enum {
    gpu, ai, llm, embeddings, agents, training,
    database, network, observability, web, personas, cloud,
};
```

Each feature provides:
- `name()` — Returns the feature name as a string
- `description()` — Returns a description of the feature
- `isCompileTimeEnabled()` — Checks if the feature was enabled at build time

## Configuration Patterns

### Creating Default Configuration

```zig
const config = Config.defaults();
```

Creates a config with all compile-time enabled features using their default settings.

### Creating Minimal Configuration

```zig
const config = Config.minimal();
```

Creates an empty config with no features enabled.

### Checking Feature Status

```zig
const config = Config.defaults();
if (config.isEnabled(.gpu)) {
    // GPU is configured
}

// Get list of enabled features
const features = try config.enabledFeatures(allocator);
defer allocator.free(features);
```

### Validation

```zig
try config.validate();  // Returns ConfigError if config violates compile-time constraints
```

## Builder Pattern

Use the fluent `Builder` for constructing configs:

```zig
var builder = Builder.init(allocator);
const config = builder
    .withDefaults()
    .withGpu(custom_gpu_config)
    .withAiDefaults()
    .withDatabase(custom_db_config)
    .build();
```

Available builder methods:
- `withDefaults()` — Enable all compile-time enabled features
- `withGpu(cfg)` / `withGpuDefaults()` — Configure GPU
- `withAi(cfg)` / `withAiDefaults()` — Configure AI
- `withLlm(cfg)` — Configure LLM (nested under AI)
- `withDatabase(cfg)` / `withDatabaseDefaults()` — Configure database
- `withNetwork(cfg)` / `withNetworkDefaults()` — Configure network
- `withObservability(cfg)` / `withObservabilityDefaults()` — Configure observability
- `withWeb(cfg)` / `withWebDefaults()` — Configure web
- `withPlugins(cfg)` — Configure plugins

## ConfigLoader

The `ConfigLoader` reads configuration from environment variables and merges with a base config:

```zig
const config_mod = @import("abi").config;

var loader = config_mod.ConfigLoader.init(allocator);
defer loader.deinit();

// Load from environment variables
const config = try loader.load();

// Or merge with a base config
const base = Config.defaults();
const config = try loader.loadWithBase(base);
```

### Environment Variable Overrides

Supported environment variables:

| Variable | Purpose |
|----------|---------|
| `ABI_GPU_BACKEND` | GPU backend (auto, cuda, vulkan, metal, none) |
| `ABI_LLM_MODEL_PATH` | Path to LLM model file |
| `ABI_LLM_TEMPERATURE` | LLM sampling temperature (0.0-1.0) |
| `ABI_LLM_MAX_TOKENS` | Maximum tokens for generation |
| `ABI_DB_PATH` | Database file path |
| `ABI_OPENAI_API_KEY` | OpenAI API credentials |
| `ABI_OLLAMA_HOST` | Ollama server address |
| `ABI_ANTHROPIC_API_KEY` | Anthropic/Claude API key |

See `loader.zig` for the complete list and implementation.

## Compile-Time Feature Control

Features can be disabled at build time, making their configs unavailable:

```bash
# Build with AI disabled
zig build -Denable-ai=false

# Build with specific GPU backend
zig build -Dgpu-backend=cuda
```

When a feature is disabled at compile time:
- Its config type is excluded from the build
- `isCompileTimeEnabled()` returns false
- `validate()` will error if you try to use a disabled feature

## Usage Example

```zig
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load configuration with environment variable overrides
    var loader = abi.config.ConfigLoader.init(allocator);
    defer loader.deinit();

    const base = abi.config.Config.defaults();
    var config = try loader.loadWithBase(base);

    // Validate against compile-time constraints
    try abi.config.validate(config);

    // Initialize framework with config
    var framework = try abi.Framework.init(allocator, config);
    defer framework.deinit();

    // Use framework...
}
```

## Module Files

| File | Purpose |
|------|---------|
| `mod.zig` | Main entry point; re-exports all configuration types |
| `gpu.zig` | GPU configuration struct and defaults |
| `ai.zig` | AI, LLM, embeddings, agents, training configurations |
| `database.zig` | Vector database configuration |
| `network.zig` | Network and distributed compute configuration |
| `observability.zig` | Metrics, tracing, monitoring configuration |
| `web.zig` | HTTP server and web utilities configuration |
| `plugin.zig` | Plugin system configuration |
| `loader.zig` | ConfigLoader for environment variable overrides |

## Notes

- All feature configs are **optional** to enable flexible feature combinations
- The builder pattern makes it easy to construct configs fluently
- Environment variables override default values via `ConfigLoader`
- Compile-time feature flags prevent configuration of disabled features
- Use `validate()` to ensure your config matches the binary's capabilities
