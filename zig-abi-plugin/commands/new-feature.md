---
name: new-feature
description: Scaffold a new ABI feature module with mod.zig, stub.zig, catalog registration, and build system wiring
argument-hint: "<feature-name>  e.g. notifications, scheduling"
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
---

# Scaffold New ABI Feature

Create a complete feature module following all 8 required steps.

## Instructions

Given the feature name argument (kebab-case), perform all steps:

### Step 1: Create mod.zig
Create `src/features/<name>/mod.zig` with a basic module structure:
```zig
const std = @import("std");
const build_options = @import("build_options");

pub const Config = struct {
    // Feature configuration
};

pub fn init(config: Config) !void {
    _ = config;
    // TODO: implement
}

pub fn deinit() void {
    // TODO: implement
}
```

### Step 2: Create matching stub.zig
Create `src/features/<name>/stub.zig` mirroring every `pub fn`:
```zig
const std = @import("std");

pub const Config = struct {};

pub fn init(config: Config) !void {
    _ = config;
    return error.FeatureDisabled;
}

pub fn deinit() void {}
```

### Step 3: Add flag to build/options.zig
Add `feat_<name>: bool = true,` to the `BuildOptions` struct. Add it to `CanonicalFlags` too.

### Step 4: Register in feature_catalog.zig
Add the feature to the `Feature` enum and the `all` array in `src/core/feature_catalog.zig`.

### Step 5: Add to test_discovery.zig
Add a test entry: `.{ .flag = "feat_<name>", .path = "features/<name>/mod.zig" }`

### Step 6: Add to flags.zig
Add `feat_<name>` field to `FlagCombo` struct if validation combos are needed.

### Step 7: Wire up in abi.zig
Add comptime feature selection in `src/abi.zig`.

### Step 8: Verify
Run `zig fmt --check` on all modified files, then report which files were created/modified.

## Important
- Use snake_case for the flag name (convert kebab-case: `my-feature` → `feat_my_feature`)
- Every pub fn in mod.zig MUST have a matching signature in stub.zig
- Stub functions return `error.FeatureDisabled`, `null`, `0`, or `void` as appropriate
