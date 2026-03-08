# config

> Unified configuration system.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-config"></a>`pub const Config`

<sup>**const**</sup> | [source](../../src/core/config/mod.zig#L75)

Unified configuration for the ABI framework.
All feature configs are optional - null means the feature is disabled.

### <a id="pub-fn-defaults-config"></a>`pub fn defaults() Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L97)

Create a config with all compile-time enabled features using defaults.

### <a id="pub-fn-minimal-config"></a>`pub fn minimal() Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L120)

Create a minimal config with no features enabled.

### <a id="pub-fn-isenabled-self-config-feature-feature-bool"></a>`pub fn isEnabled(self: Config, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L125)

Check if a feature is enabled in this config.

### <a id="pub-fn-enabledfeatures-self-config-allocator-std-mem-allocator-feature"></a>`pub fn enabledFeatures(self: Config, allocator: std.mem.Allocator) ![]Feature`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L158)

Get list of enabled features.

### <a id="pub-const-builder"></a>`pub const Builder`

<sup>**const**</sup> | [source](../../src/core/config/mod.zig#L178)

Fluent builder for constructing Config.

### <a id="pub-fn-with-self-builder-comptime-feature-feature-cfg-anytype-builder"></a>`pub fn with(self: *Builder, comptime feature: Feature, cfg: anytype) *Builder`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L229)

Enable a feature with explicit configuration.

### <a id="pub-fn-withdefault-self-builder-comptime-feature-feature-builder"></a>`pub fn withDefault(self: *Builder, comptime feature: Feature) *Builder`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L251)

Enable a feature with its default configuration.

### <a id="pub-fn-build-self-builder-config"></a>`pub fn build(self: *Builder) Config`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L273)

Finalize and return the built config; no allocation.

### <a id="pub-fn-validate-cfg-config-configerror-void"></a>`pub fn validate(cfg: Config) ConfigError!void`

<sup>**fn**</sup> | [source](../../src/core/config/mod.zig#L295)

Validate configuration against compile-time constraints.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use the `$zig-master` Codex skill for ABI Zig validation, docs generation, and build-wiring changes.
