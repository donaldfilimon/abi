# Registry Module

The feature registry system provides a unified interface for feature registration and lifecycle management.

## Overview

The registry supports three registration modes for different use cases:

- **Comptime-only**: Zero overhead, features resolved at compile time
- **Runtime-toggle**: Compiled in but can be enabled/disabled at runtime
- **Dynamic**: Features loaded from shared libraries (planned for future)

## Basic Usage

```zig
const registry = @import("abi").registry;

var reg = registry.Registry.init(allocator);
defer reg.deinit();

// Register features
try reg.registerComptime(.gpu);
try reg.registerRuntimeToggle(.ai, AiContext, &ai_config);

// Query features
if (reg.isEnabled(.gpu)) {
    // Use GPU feature
}

// Enable/disable at runtime (runtime-toggle only)
try reg.enableFeature(.ai);
```

## Key Types

### Feature

An enum representing available features in the framework (e.g., `.gpu`, `.ai`, `.database`, `.network`, `.llm`, `.embeddings`, `.agents`, `.training`).

### Registry

The central registry managing feature lifecycle. Maintains registrations and runtime toggles.

**Key Methods:**
- `registerComptime(feature)` - Register a zero-overhead compile-time feature
- `registerRuntimeToggle(feature, ContextType, config_ptr)` - Register a feature that can be toggled at runtime
- `enableFeature(feature)` / `disableFeature(feature)` - Toggle runtime features
- `initFeature(feature)` / `deinitFeature(feature)` - Manage feature lifecycle
- `isEnabled(feature)` - Check if feature is currently available
- `isRegistered(feature)` - Check if feature is registered
- `isInitialized(feature)` - Check if feature is initialized and ready

### RegistrationMode

Indicates how a feature is registered:

- `.comptime_only` - Always enabled if registered (zero cost)
- `.runtime_toggle` - Can be toggled at runtime via `enableFeature`/`disableFeature`
- `.dynamic` - Loaded from shared library (future)

## Registration Modes

### Comptime-Only

Use for features that are either always available or not needed:

```zig
try reg.registerComptime(.gpu);
// Feature is now always enabled (assuming compiled in with -Denable-gpu=true)
```

Provides zero runtime overhead since the feature state is known at compile time.

### Runtime-Toggle

Use for features that should be dynamically enabled/disabled:

```zig
try reg.registerRuntimeToggle(.ai, AiContext, &ai_config);

// Later, enable or disable at runtime
try reg.enableFeature(.ai);      // Initialize and enable
try reg.disableFeature(.ai);     // Disable and clean up

// Get access to the context
const ai = try reg.getContext(.ai, AiContext);
```

Runtime-toggle features:
- Start disabled by default
- Must be enabled before use
- Can be toggled on/off without restart
- Automatically deinitialize when disabled

### Dynamic (Planned)

Reserved for future plugin loading from shared libraries.

## Sub-modules

- **`types.zig`** - Core types (`Feature`, `RegistrationMode`, `FeatureRegistration`, `Error`)
- **`registration.zig`** - Registration functions for different modes
- **`lifecycle.zig`** - Feature initialization, shutdown, and state management

## Error Handling

The registry uses a `Registry.Error` enum:

- `FeatureNotRegistered` - Feature not found in registry
- `FeatureDisabled` - Feature is disabled at runtime
- `FeatureAlreadyRegistered` - Feature already registered
- `NotInitialized` - Feature not yet initialized
- `InvalidMode` - Operation not valid for feature's registration mode
- `FeatureNotCompiledIn` - Feature not included in build (check build flags)

## Build Integration

Features must be compiled in to be registered. Check with build flags:

```bash
zig build -Denable-gpu=true -Denable-ai=true
```

At runtime, use `comptime` checks to verify:

```zig
if (comptime isFeatureCompiledIn(.gpu)) {
    try reg.registerComptime(.gpu);
}
```

## Feature Dependencies

Some features have parent dependencies:

- `.llm`, `.embeddings`, `.agents`, `.training` require `.ai` parent
- `.training` also requires `.ai`

Query parent with `getParentFeature(feature)`.
