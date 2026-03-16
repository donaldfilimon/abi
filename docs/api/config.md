---
title: config API
purpose: Generated API reference for config
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# config

> Configuration Module

Re-exports all configuration types from domain-specific files.
Import this module for access to all configuration types.

Use `ConfigLoader` (see `loader.zig`) to load config from environment variables
(e.g. `ABI_GPU_BACKEND`, `ABI_LLM_MODEL_PATH`). Use `Config.Builder` for fluent construction.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-config"></a>`pub const Config`

<sup>**const**</sup> | [source](../../src/core/config/mod.zig#L76)

Unified configuration for the ABI framework.
All feature configs are optional - null means the feature is disabled.

### <a id="pub-fn-defaults-config"></a>`pub fn defaults() Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L98)

Create a config with all compile-time enabled features using defaults.

### <a id="pub-fn-minimal-config"></a>`pub fn minimal() Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L121)

Create a minimal config with no features enabled.

### <a id="pub-fn-isenabled-self-config-feature-feature-bool"></a>`pub fn isEnabled(self: Config, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L126)

Check if a feature is enabled in this config.

### <a id="pub-fn-enabledfeatures-self-config-allocator-std-mem-allocator-feature"></a>`pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L159)

Get list of enabled features.

### <a id="pub-const-builder"></a>`pub const Builder`

<sup>**const**</sup> | [source](../../src/core/config/mod.zig#L179)

Fluent builder for constructing Config.

### <a id="pub-fn-with-self-builder-comptime-feature-feature-cfg-anytype-builder"></a>`pub fn with(self: *Builder, comptime feature: Feature, cfg: anytype) *Builder`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L230)

Enable a feature with explicit configuration.

### <a id="pub-fn-withdefault-self-builder-comptime-feature-feature-builder"></a>`pub fn withDefault(self: *Builder, comptime feature: Feature) *Builder`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L252)

Enable a feature with its default configuration.

### <a id="pub-fn-build-self-builder-config"></a>`pub fn build(self: *Builder) Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L274)

Finalize and return the built config; no allocation.

### <a id="pub-fn-validate-cfg-config-configerror-void"></a>`pub fn validate(cfg: Config) ConfigError!void`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L296)

Validate configuration against compile-time constraints.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
