# registry API Reference

> Plugin registry (comptime, runtime, dynamic)

**Source:** [`src/registry/mod.zig`](../../src/registry/mod.zig)

---

Feature Registry System

Provides a unified interface for feature registration and lifecycle management
supporting three registration modes:

- **Comptime-only**: Zero overhead, features resolved at compile time
- **Runtime-toggle**: Compiled in but can be enabled/disabled at runtime
- **Dynamic**: Features loaded from shared libraries at runtime (future)

## Usage

```zig
const registry = @import("registry/mod.zig");

var reg = registry.Registry.init(allocator);
defer reg.deinit();

// Register features
try reg.registerComptime(.gpu);
try reg.registerRuntimeToggle(.ai, ai_mod.Context, &ai_config);

// Query features
if (reg.isEnabled(.gpu)) {
// Use GPU...
}
```

---

## API

### `pub const RegistrationMode`

<sup>**type**</sup>

Registration mode determines how features are discovered and managed.

### `pub const FeatureRegistration`

<sup>**type**</sup>

Feature registration entry with lifecycle management.

### `pub const Registry`

<sup>**type**</sup>

Central registry managing feature lifecycle across all registration modes.

### `pub fn init(allocator: std.mem.Allocator) Registry`

<sup>**fn**</sup>

Initialize empty registry.

### `pub fn deinit(self: *Registry) void`

<sup>**fn**</sup>

Cleanup all registered features and plugin state.

### `pub fn registerComptime(self: *Registry, comptime feature: Feature) Error!void`

<sup>**fn**</sup>

Register a feature for comptime-only resolution.
The feature must be enabled at compile time via build_options.
This is zero-overhead - just validates feature exists at comptime.

### `pub fn registerRuntimeToggle(`

<sup>**fn**</sup>

Register a feature with runtime toggle capability.
Feature must be compiled in, but can be enabled/disabled at runtime.

### `pub fn registerDynamic(`

<sup>**fn**</sup>

Register a feature for dynamic loading from a shared library (future).

### `pub fn initFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup>

Initialize a registered feature. For runtime_toggle and dynamic modes.

### `pub fn deinitFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup>

Shutdown a feature, releasing resources.

### `pub fn isRegistered(self: *const Registry, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is registered.

### `pub fn isEnabled(self: *const Registry, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is currently enabled.
For comptime_only: always true if registered
For runtime_toggle/dynamic: depends on runtime state

### `pub fn isInitialized(self: *const Registry, feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is initialized and ready to use.

### `pub fn getMode(self: *const Registry, feature: Feature) ?RegistrationMode`

<sup>**fn**</sup>

Get the registration mode for a feature.

### `pub fn getContext(`

<sup>**fn**</sup>

Get the context for a feature. Returns error if not initialized.

### `pub fn enableFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup>

Enable a runtime-toggleable feature.

### `pub fn disableFeature(self: *Registry, feature: Feature) Error!void`

<sup>**fn**</sup>

Disable a runtime-toggleable feature. Deinitializes if currently initialized.

### `pub fn listFeatures(self: *const Registry, allocator: std.mem.Allocator) Error![]Feature`

<sup>**fn**</sup>

Get list of all registered features.

### `pub fn count(self: *const Registry) usize`

<sup>**fn**</sup>

Get count of registered features.

### `pub fn isFeatureCompiledIn(comptime feature: Feature) bool`

<sup>**fn**</sup>

Check if a feature is compiled in via build_options.

### `pub fn getParentFeature(feature: Feature) ?Feature`

<sup>**fn**</sup>

Get parent feature for sub-features.

---

*Generated automatically by `zig build gendocs`*
