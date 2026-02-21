# config

> migration. Compatibility is preserved for one release cycle.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

**Availability:** Always enabled

---

Configuration Module

Re-exports all configuration types from domain-specific files.
Import this module for access to all configuration types.

Use `ConfigLoader` (see `loader.zig`) to load config from environment variables
(e.g. `ABI_GPU_BACKEND`, `ABI_LLM_MODEL_PATH`). Use `Config.Builder` for fluent construction.

---

## API

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

### `pub fn build(self: *Builder) Config`

<sup>**fn**</sup>

Finalize and return the built config; no allocation.

### `pub fn validate(cfg: Config) ConfigError!void`

<sup>**fn**</sup>

Validate configuration against compile-time constraints.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
