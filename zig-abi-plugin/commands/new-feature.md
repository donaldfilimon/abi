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

Create a complete feature module following all 10 required steps.

## Instructions

Given the feature name argument (kebab-case), perform all steps:

### Step 1: Create types.zig
Create `src/features/<name>/types.zig` with shared type definitions used by both mod and stub. Per the mod/stub contract, shared types go in `types.zig` so both files stay in sync automatically:
```zig
//! Shared types for the <name> feature.
//!
//! Both mod.zig and stub.zig import from this file so that
//! public type signatures stay in sync automatically.

const std = @import("std");

/// Configuration for the <name> feature.
pub const Config = struct {
    // TODO: add feature-specific configuration fields
};

// TODO: add shared error sets, enums, and structs used in public API signatures
```

### Step 2: Create mod.zig
Create `src/features/<name>/mod.zig` importing shared types from types.zig:
```zig
//! <Name> Feature Module
//!
//! TODO: describe what this feature provides.

const std = @import("std");
const build_options = @import("build_options");
const types = @import("types.zig");

pub const Config = types.Config;

pub fn init(config: Config) !void {
    _ = config;
    // TODO: implement
}

pub fn deinit() void {
    // TODO: implement
}

test {
    std.testing.refAllDecls(@This());
}
```

### Step 3: Create matching stub.zig
Create `src/features/<name>/stub.zig` mirroring every `pub fn` and importing shared types:
```zig
//! <Name> stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

pub const Config = types.Config;

pub fn init(config: Config) !void {
    _ = config;
    return error.FeatureDisabled;
}

pub fn deinit() void {}

test {
    std.testing.refAllDecls(@This());
}
```

### Step 4: Add flag to build/options.zig
Add `feat_<name>: bool = true,` to the `BuildOptions` struct (alphabetical order within the feat_* block). The comptime validation at the bottom of options.zig ensures BuildOptions stays in sync with feature_catalog.zig — a missing flag will be a compile error.

### Step 5: Register in feature_catalog.zig
Add the feature to the `Feature` enum in `src/core/feature_catalog.zig`, and add a metadata entry to the `all` array with:
- `.feature` — enum value
- `.description` — one-line purpose
- `.compile_flag_field` — `"feat_<name>"`
- `.parity_spec` — ParitySpec enum value (e.g., `.ai`, `.database`, or create new)
- `.parent` — parent feature or `null` (e.g., `.ai` for AI subfeatures)
- `.real_module_path` — `"features/<name>/mod.zig"`
- `.stub_module_path` — `"features/<name>/stub.zig"`

### Step 6: Add to module_catalog.zig
Add a test entry to `build/module_catalog.zig` manifest:
```zig
.{ .flag = "feat_<name>", .path = "features/<name>/mod.zig" },
```

### Step 7: Add to flags.zig
Two changes in `build/flags.zig`:
1. Add `feat_<name>: bool = true,` field to the `FlagCombo` struct
2. Add validation rows to `validation_matrix`:
   - `<name>-only` row (only this feature enabled, rest false)
   - `no-<name>` row (this feature false, rest true)

### Step 8: Wire up in root.zig
Add comptime feature selection in `src/root.zig`:
```zig
pub const <name> = if (build_options.feat_<name>)
    @import("features/<name>/mod.zig")
else
    @import("features/<name>/stub.zig");
```

### Step 9: Verify
Run `zig fmt --check` on all modified files, then report:
- Files created/modified
- Which validation steps still need to run (validate-flags, full-check)
- On Darwin: note that full validation may still require Linux CI or another host with a working Zig linker

### Step 10: Wire into Framework Lifecycle

Add the new feature's comptime-gated import to `src/core/framework/feature_imports.zig`:

```zig
pub const <name> = if (build_options.feat_<name>) @import("../../features/<name>/mod.zig") else @import("../../features/<name>/stub.zig");
```

This file is the central hub where the framework lifecycle (init, shutdown, health checks) discovers features. Without this step, the feature will compile but won't participate in `App.init()` / `App.deinit()`.

## Important
- Use snake_case for the flag name (convert kebab-case: `my-feature` → `feat_my_feature`)
- Every pub fn in mod.zig MUST have a matching signature in stub.zig
- Stub functions return `error.FeatureDisabled`, `null`, `0`, or `void` as appropriate
- Both mod.zig and stub.zig should have `test { std.testing.refAllDecls(@This()); }`
- Named module imports: use `@import("build_options")` for build options — all other imports within `src/` must use relative paths
