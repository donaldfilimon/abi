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
//! <Name> Feature Module
//!
//! TODO: describe what this feature provides.

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

test {
    std.testing.refAllDecls(@This());
}
```

### Step 2: Create matching stub.zig
Create `src/features/<name>/stub.zig` mirroring every `pub fn`:
```zig
//! <Name> stub — disabled at compile time.

const std = @import("std");

pub const Config = struct {};

pub fn init(config: Config) !void {
    _ = config;
    return error.FeatureDisabled;
}

pub fn deinit() void {}

test {
    std.testing.refAllDecls(@This());
}
```

### Step 3: Add flag to build/options.zig
Add `feat_<name>: bool = true,` to the `BuildOptions` struct (alphabetical order within the feat_* block). The comptime validation at the bottom of options.zig ensures BuildOptions stays in sync with feature_catalog.zig — a missing flag will be a compile error.

### Step 4: Register in feature_catalog.zig
Add the feature to the `Feature` enum in `src/core/feature_catalog.zig`, and add a metadata entry to the `all` array with:
- `.feature` — enum value
- `.description` — one-line purpose
- `.compile_flag_field` — `"feat_<name>"`
- `.parity_spec` — ParitySpec enum value (e.g., `.ai`, `.database`, or create new)
- `.parent` — parent feature or `null` (e.g., `.ai` for AI subfeatures)
- `.real_module_path` — `"features/<name>/mod.zig"`
- `.stub_module_path` — `"features/<name>/stub.zig"`

### Step 5: Add to test_discovery.zig
Add a test entry to `build/test_discovery.zig` manifest:
```zig
.{ .flag = "feat_<name>", .path = "features/<name>/mod.zig" },
```

### Step 6: Add to flags.zig
Two changes in `build/flags.zig`:
1. Add `feat_<name>: bool = true,` field to the `FlagCombo` struct
2. Add validation rows to `validation_matrix`:
   - `<name>-only` row (only this feature enabled, rest false)
   - `no-<name>` row (this feature false, rest true)

### Step 7: Wire up in root.zig
Add comptime feature selection in `src/root.zig`:
```zig
pub const <name> = if (build_options.feat_<name>)
    @import("features/<name>/mod.zig")
else
    @import("features/<name>/stub.zig");
```

### Step 8: Verify
Run `zig fmt --check` on all modified files, then report:
- Files created/modified
- Which validation steps still need to run (validate-flags, full-check)
- On Darwin: note that full validation may still require Linux CI or another host with a working Zig linker

## Important
- Use snake_case for the flag name (convert kebab-case: `my-feature` → `feat_my_feature`)
- Every pub fn in mod.zig MUST have a matching signature in stub.zig
- Stub functions return `error.FeatureDisabled`, `null`, `0`, or `void` as appropriate
- Both mod.zig and stub.zig should have `test { std.testing.refAllDecls(@This()); }`
- Named module imports: use `@import("wdbx")`, `@import("build_options")` — never relative paths to named module roots
