# runtime-engine API Reference

> Work-stealing task execution engine

**Source:** [`src/runtime/engine/mod.zig`](../../src/runtime/engine/mod.zig)

---

Task Execution Engine

This module provides the core task execution engine with:

- `Engine` - Work-stealing distributed compute engine
- `EngineConfig` - Engine configuration options
- NUMA-aware task scheduling
- Result handling and task lifecycle
- `ResultCache` - High-performance result caching for fast-path completion

Note: On WASM/freestanding targets without thread support, the engine
provides stub implementations that return appropriate errors.

---

## API

### `pub fn createEngine(allocator: std.mem.Allocator) !Engine`

<sup>**fn**</sup>

Create an engine with default configuration.

### `pub fn createEngineWithConfig(allocator: std.mem.Allocator, config: EngineConfig) !Engine`

<sup>**fn**</sup>

Create an engine with custom configuration.

---

*Generated automatically by `zig build gendocs`*
