# registry

> Plugin registry for feature management.

**Source:** [`src/core/registry/mod.zig`](../../src/core/registry/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-registry"></a>`pub const Registry`

<sup>**const**</sup> | [source](../../src/core/registry/mod.zig#L46)

Central registry managing feature lifecycle across all registration modes.

### <a id="pub-const-error"></a>`pub const Error`

<sup>**const**</sup> | [source](../../src/core/registry/mod.zig#L48)

Error type for Registry operations (alias for backward compatibility).

### <a id="pub-fn-init-allocator-std-mem-allocator-registry"></a>`pub fn init(allocator: std.mem.Allocator) Registry`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L59)

Initialize empty registry.

### <a id="pub-fn-deinit-self-registry-void"></a>`pub fn deinit(self: *Registry) void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L68)

Cleanup all registered features and plugin state.

### <a id="pub-fn-registercomptime-self-registry-comptime-feature-feature-error-void"></a>`pub fn registerComptime(self: *Registry, comptime feature: Feature) Error!void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L92)

Register a feature for comptime-only resolution.
The feature must be enabled at compile time via build_options.
This is zero-overhead - just validates feature exists at comptime.

### <a id="pub-fn-registerruntimetoggle-self-registry-comptime-feature-feature-comptime-contexttype-type-config-ptr-const-anyopaque-error-void"></a>`pub fn registerRuntimeToggle( self: *Registry, comptime feature: Feature, comptime ContextType: type, config_ptr: *const anyopaque, ) Error!void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L98)

Register a feature with runtime toggle capability.
Feature must be compiled in, but can be enabled/disabled at runtime.

### <a id="pub-fn-registerdynamic-self-registry-feature-feature-library-path-const-u8-error-void"></a>`pub fn registerDynamic( self: *Registry, feature: Feature, library_path: []const u8, ) Error!void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L108)

Register a feature for dynamic loading from a shared library (future).

### <a id="pub-fn-initfeature-self-registry-feature-feature-error-void"></a>`pub fn initFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L121)

Initialize a registered feature. For runtime_toggle and dynamic modes.

### <a id="pub-fn-deinitfeature-self-registry-feature-feature-error-void"></a>`pub fn deinitFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L126)

Shutdown a feature, releasing resources.

### <a id="pub-fn-enablefeature-self-registry-feature-feature-error-void"></a>`pub fn enableFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L131)

Enable a runtime-toggleable feature.

### <a id="pub-fn-disablefeature-self-registry-feature-feature-error-void"></a>`pub fn disableFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L136)

Disable a runtime-toggleable feature. Deinitializes if currently initialized.

### <a id="pub-fn-isregistered-self-const-registry-feature-feature-bool"></a>`pub fn isRegistered(self: *const Registry, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L145)

Check if a feature is registered.

### <a id="pub-fn-isenabled-self-const-registry-feature-feature-bool"></a>`pub fn isEnabled(self: *const Registry, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L152)

Check if a feature is currently enabled.
For comptime_only: always true if registered
For runtime_toggle/dynamic: depends on runtime state

### <a id="pub-fn-isinitialized-self-const-registry-feature-feature-bool"></a>`pub fn isInitialized(self: *const Registry, feature: Feature) bool`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L169)

Check if a feature is initialized and ready to use.

### <a id="pub-fn-getmode-self-const-registry-feature-feature-registrationmode"></a>`pub fn getMode(self: *const Registry, feature: Feature) ?RegistrationMode`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L177)

Get the registration mode for a feature.

### <a id="pub-fn-getcontext-self-const-registry-feature-feature-comptime-contexttype-type-error-contexttype"></a>`pub fn getContext( self: *const Registry, feature: Feature, comptime ContextType: type, ) Error!*ContextType`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L185)

Get the context for a feature. Returns error if not initialized.

### <a id="pub-fn-listfeatures-self-const-registry-allocator-std-mem-allocator-error-feature"></a>`pub fn listFeatures(self: *const Registry, allocator: std.mem.Allocator) Error![]Feature`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L200)

Get list of all registered features.

### <a id="pub-fn-count-self-const-registry-usize"></a>`pub fn count(self: *const Registry) usize`

<sup>**fn**</sup> | [source](../../src/core/registry/mod.zig#L213)

Get count of registered features.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
