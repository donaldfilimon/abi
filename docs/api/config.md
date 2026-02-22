# config

> Unified configuration system.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-config"></a>`pub const Config`

<sup>**const**</sup> | [source](../../src/core/config/mod.zig#L79)

Unified configuration for the ABI framework.
All feature configs are optional - null means the feature is disabled.

### <a id="pub-fn-defaults-config"></a>`pub fn defaults() Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L101)

Create a config with all compile-time enabled features using defaults.

### <a id="pub-fn-minimal-config"></a>`pub fn minimal() Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L124)

Create a minimal config with no features enabled.

### <a id="pub-fn-isenabled-self-config-feature-feature-bool"></a>`pub fn isEnabled(self: Config, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L129)

Check if a feature is enabled in this config.

### <a id="pub-fn-enabledfeatures-self-config-allocator-std-mem-allocator-feature"></a>`pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L159)

Get list of enabled features.

### <a id="pub-const-builder"></a>`pub const Builder`

<sup>**const**</sup> | [source](../../src/core/config/mod.zig#L179)

Fluent builder for constructing Config.

### <a id="pub-fn-build-self-builder-config"></a>`pub fn build(self: *Builder) Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L391)

Finalize and return the built config; no allocation.

### <a id="pub-fn-validate-cfg-config-configerror-void"></a>`pub fn validate(cfg: Config) ConfigError!void`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L413)

Validate configuration against compile-time constraints.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
