# Abi AI Framework Plugin System

## Overview

The plugin system provides a stable Application Binary Interface (ABI) for extending the Abi AI Framework with custom functionality. Plugins can be written in any language that supports C ABI.

## Quick Start

### Using Plugins

```zig
const plugins = @import("plugins");

// Initialize plugin system
var registry = try plugins.Registry.init(allocator);
defer registry.deinit();

// Load a plugin
try registry.loadPlugin("./plugins/my_plugin.so");

// Get and use the plugin
const plugin = try registry.getPlugin("my_plugin");
try plugin.start();
```

### Creating Plugins

See the [Plugin Development Guide](../../docs/PLUGIN_DEVELOPMENT_GUIDE.md) for detailed instructions.

## Core Components

### 1. Plugin Interface (`interface.zig`)

Defines the C-compatible plugin interface:
- Required functions: `get_info`, `init`, `deinit`
- Optional functions: lifecycle, processing, configuration, etc.
- Entry point: `abi_plugin_create`

### 2. Plugin Types (`types.zig`)

Core types and structures:
- `PluginInfo` - Plugin metadata
- `PluginVersion` - Version management
- `PluginConfig` - Configuration
- `PluginContext` - Runtime context
- `PluginState` - State management
- `PluginError` - Error types

### 3. Plugin Loader (`loader.zig`)

Handles dynamic loading:
- Platform-specific library loading
- Symbol resolution
- Version checking
- Dependency management

### 4. Plugin Registry (`registry.zig`)

Manages loaded plugins:
- Registration and discovery
- Lifecycle management
- Event distribution
- Service lookup

## ABI Stability

Current ABI Version: **1.0.0**

The plugin system guarantees:
- Binary compatibility within major versions
- C calling convention for cross-language support
- Stable structure layouts for `extern` types
- Forward compatibility through optional functions

## Plugin Types

- **Database**: `vector_database`, `indexing_algorithm`, `compression_algorithm`
- **AI/ML**: `neural_network`, `embedding_generator`, `training_algorithm`, `inference_engine`
- **Processing**: `text_processor`, `image_processor`, `audio_processor`, `data_transformer`
- **I/O**: `data_loader`, `data_exporter`, `protocol_handler`
- **Utility**: `logger`, `metrics_collector`, `security_provider`, `configuration_provider`
- **Custom**: Domain-specific plugins

## Examples

- [Basic Plugin](../../examples/plugins/example_plugin.zig) - Minimal implementation
- [Advanced Plugin](../../examples/plugins/advanced_plugin_example.zig) - Full-featured example
- [Discord Plugin](../../examples/plugins/discord_plugin.zig) - Real-world integration

## Testing

Run plugin tests:
```bash
zig build test-plugins
zig build test-abi
```

Validate a plugin:
```bash
./plugin_validator my_plugin.so --verbose
```

Test harness:
```bash
./plugin_test_harness my_plugin.so --stress --benchmark
```

## Documentation

- [Plugin ABI Specification](../../docs/PLUGIN_ABI_SPECIFICATION.md)
- [Plugin Development Guide](../../docs/PLUGIN_DEVELOPMENT_GUIDE.md)
- [Plugin System Documentation](../../docs/PLUGIN_SYSTEM.md)

## Best Practices

1. **Memory Management**: Always free allocations in `deinit`
2. **Error Handling**: Return appropriate error codes
3. **Thread Safety**: Use synchronization for shared state
4. **Performance**: Profile and optimize hot paths
5. **Logging**: Use the provided logging interface

## Troubleshooting

Common issues:
- Plugin fails to load: Check ABI version and dependencies
- Crashes: Verify memory alignment and null checks
- Memory leaks: Ensure proper cleanup in deinit
- Performance: Profile with appropriate tools

## Contributing

When modifying the plugin system:
1. Maintain ABI compatibility
2. Update version for breaking changes
3. Add tests for new features
4. Update documentation
5. Test cross-platform compatibility