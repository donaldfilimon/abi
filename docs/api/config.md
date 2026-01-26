# config API Reference

> Unified configuration system with builder pattern

**Source:** [`src/config/mod.zig`](../../src/config/mod.zig)

---

Configuration Module

Re-exports all configuration types from domain-specific files.
Import this module for access to all configuration types.

---

## API

### `pub const Feature`

<sup>**type**</sup>

Available features in the framework.

### `pub const Config`

<sup>**type**</sup>

Unified configuration for the ABI framework.
All feature configs are optional - null means the feature is disabled.

### `pub fn defaults() Config`

<sup>**fn**</sup>

Create a config with all compile-time enabled features using defaults.

### `pub fn minimal() Config`

<sup>**fn**</sup>

Create a minimal config with no features enabled.

### `pub fn isEnabled(self: Config, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is enabled in this config.

### `pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature`

<sup>**fn**</sup>

Get list of enabled features.

### `pub const Builder`

<sup>**type**</sup>

Fluent builder for constructing Config.

### `pub fn validate(config: Config) ConfigError!void`

<sup>**fn**</sup>

Validate configuration against compile-time constraints.

---

*Generated automatically by `zig build gendocs`*
