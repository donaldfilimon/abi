---
name: feature-scaffolding
description: This skill provides the complete procedure for adding a new comptime-gated feature to the ABI Zig framework. Use when the user asks to add a new feature, scaffold a feature module, create mod.zig/stub.zig/types.zig, wire a new feature flag, or asks "how do I add a feature". Triggers on "new feature", "add feature", "scaffold feature", "feature gate", "mod.zig stub.zig", or "/zig-abi:new-feature".
---

# Adding a Comptime-Gated Feature to ABI

## Overview

Each ABI feature lives in `src/features/<name>/` with three files (`mod.zig`, `stub.zig`, `types.zig`) and is wired through seven build/catalog locations. The procedure below adds a feature called `<name>` gated by flag `feat_<name>`.

## Prerequisites

Confirm the feature name is not already taken. List existing directories:

```bash
ls src/features/
```

There are currently 19 feature directories. Choose a `lower_snake_case` name that does not collide.

## Step 1: Create the Feature Directory

Create three files under `src/features/<name>/`.

### types.zig

Shared types imported by both mod and stub. Never duplicate type definitions across mod/stub.

```zig
//! Shared types for the <name> feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");

/// Errors returned by <name> operations.
pub const <Name>Error = error{
    FeatureDisabled,
    OutOfMemory,
};

pub const Error = <Name>Error;
```

### mod.zig

Real implementation, compiled when `feat_<name> = true`.

```zig
//! <Name> Feature
//!
//! <One-line description of what this feature provides.>

pub const types = @import("types.zig");

const std = @import("std");

pub const <Name>Error = types.<Name>Error;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = true };
    }

    pub fn deinit(self: *Context) void {
        self.initialized = false;
    }
};

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
```

### stub.zig

No-op implementation, compiled when `feat_<name> = false`. Must match mod.zig public signatures exactly.

```zig
//! <Name> stub -- disabled at compile time.

const std = @import("std");
pub const types = @import("types.zig");

pub const <Name>Error = types.<Name>Error;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = false };
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
```

For features that accept a config struct on `init`, use `StubFeature` or `StubFeatureNoConfig` from `core/stub_context.zig` instead of hand-writing the Context. Import it via relative path: `@import("../../core/stub_context.zig")`.

If mod.zig re-exports sub-modules (e.g., `pub const foo = @import("foo.zig")`), stub.zig must re-export matching empty structs (`pub const foo = struct {};`).

## Step 2: Add the Feature Flag

Edit `build/options.zig`. Add `feat_<name>: bool,` to both `CanonicalFlags` and `BuildOptions` structs, alphabetically among the existing flags (or at the end of the feature block).

## Step 3: Add to FlagCombo

Edit `build/flags.zig`. Add `feat_<name>: bool = false,` to the `FlagCombo` struct alongside the other flags. The comptime validation block at the bottom of the file automatically checks that every catalog flag has a corresponding FlagCombo field.

## Step 4: Add to Feature Catalog

Edit `src/core/feature_catalog.zig`. Three additions required:

1. Add the variant to the `Feature` enum.
2. Add the variant to the `ParitySpec` enum.
3. Add a catalog entry to the `all` array:

```zig
.{
    .feature = .<name>,
    .description = "<Short description>",
    .compile_flag_field = "feat_<name>",
    .parity_spec = .<name>,
    .real_module_path = "features/<name>/mod.zig",
    .stub_module_path = "features/<name>/stub.zig",
},
```

Place it before the non-feature entries (lsp, mcp) to keep feature-gated items grouped.

## Step 5: Add Conditional Import to root.zig

Edit `src/root.zig`. Add in the "Features (comptime-gated mod/stub)" section:

```zig
/// <Short description>.
pub const <name> = if (build_options.feat_<name>) @import("features/<name>/mod.zig") else @import("features/<name>/stub.zig");
```

Place it alphabetically or at the end of the feature block, matching the existing style with a doc comment.

## Step 6: Add to feature_imports.zig

Edit `src/core/framework/feature_imports.zig`. Add:

```zig
pub const <name>_mod = if (build_options.feat_<name>) @import("../../features/<name>/mod.zig") else @import("../../features/<name>/stub.zig");
```

This centralizes the import so `framework.zig`, `context_init.zig`, and `shutdown.zig` pick it up automatically.

## Step 7: Add to module_catalog.zig

Edit `build/module_catalog.zig`. Two additions:

### Gendocs entry

Add to the `public_modules` array:

```zig
.{ .name = "<name>", .path = "src/features/<name>/mod.zig", .description = "<Short description>", .build_flag = "feat_<name>" },
```

### Feature test entry

Add to the `feature_test_manifest` array:

```zig
.{ .flag = "feat_<name>", .path = "src/features/<name>/mod.zig" },
```

Add additional entries if the feature has sub-modules with their own test blocks.

## Step 8: Update the Build Options Stub

Edit `tools/cli/tests/build_options_stub.zig`. Add:

```zig
pub const feat_<name> = true;
```

Set to `true` unless the feature should default to disabled (like `feat_mobile`).

## Step 9: Wire Flag Registration in build.zig

Locate the section in `build.zig` where feature flags are registered with `b.option(bool, "feat-<name>", ...)`. Add:

```zig
const feat_<name> = b.option(bool, "feat-<name>", "<Description>") orelse true;
```

Use `orelse true` for features enabled by default, `orelse false` for opt-in features.

Pass the flag into the build options struct and into `canonicalToBuildOptions` / wherever the flags are aggregated (follow the pattern of neighboring flags in the same function).

## Step 10: Format and Verify

Run the following verification gates in order:

```bash
# 1. Format check (always works, even without host Zig)
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/

# 2. Validate all 58 flag combos still compile
zig build validate-flags

# 3. Verify mod/stub public signature parity
zig build check-stub-parity

# 4. Regenerate documentation artifacts
zig build gendocs

# 5. Refresh CLI registry if adding CLI commands
zig build refresh-cli-registry

# 6. Full pre-commit gate
zig build full-check
```

On Darwin 25+, ensure a host-built Zig matching `.zigversion` is on PATH for steps that require linking.

## Checklist

Use this list to confirm completeness before committing:

- [ ] `src/features/<name>/types.zig` created with shared error set
- [ ] `src/features/<name>/mod.zig` created with `isEnabled() -> true`
- [ ] `src/features/<name>/stub.zig` created with `isEnabled() -> false`, signatures match mod
- [ ] `build/options.zig` `CanonicalFlags` has `feat_<name>: bool`
- [ ] `build/options.zig` `BuildOptions` has `feat_<name>: bool`
- [ ] `build/flags.zig` `FlagCombo` has `feat_<name>: bool = false`
- [ ] `src/core/feature_catalog.zig` `Feature` enum has `.<name>`
- [ ] `src/core/feature_catalog.zig` `ParitySpec` enum has `.<name>`
- [ ] `src/core/feature_catalog.zig` `all` array has catalog entry
- [ ] `src/root.zig` has conditional import line
- [ ] `src/core/framework/feature_imports.zig` has `<name>_mod` import
- [ ] `build/module_catalog.zig` `public_modules` has gendocs entry
- [ ] `build/module_catalog.zig` `feature_test_manifest` has test entry
- [ ] `tools/cli/tests/build_options_stub.zig` has `feat_<name>` constant
- [ ] `build.zig` registers the flag with `b.option` and passes it through
- [ ] `zig fmt --check` passes
- [ ] `zig build validate-flags` passes
- [ ] `zig build check-stub-parity` passes
- [ ] `zig build full-check` passes

## Common Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Stub signature differs from mod | Compile error when feature disabled | Copy mod.zig public API, replace bodies with no-ops |
| Missing from feature_catalog.zig | Comptime error in flags.zig validation | Catalog entry is required before flag validation passes |
| Using `@import("abi")` inside src/ | Circular import error | Use relative paths within the `abi` module |
| Forgetting build_options_stub.zig | CLI standalone test matrix fails | Always mirror new flags in the stub |
| Sub-module in mod.zig not in stub | Compile error when disabled code references it | Add `pub const foo = struct {};` in stub |
| Skipping `zig build gendocs` | Docs consistency check fails in full-check | Always regenerate after catalog changes |
