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

For features that accept a config struct on `init`, use `StubFeature` or `StubFeatureNoConfig` from `core/stub_helpers.zig` instead of hand-writing the Context:
```zig
const stub_helpers = @import("../../core/stub_helpers.zig");
const feature = stub_helpers.StubFeature(MyConfig, MyError);
pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;
```

If mod.zig re-exports sub-modules (e.g., `pub const tokenizer = @import("tokenizer.zig")`), stub.zig must re-export matching stubs with the same declaration name:
```zig
pub const tokenizer = struct {
    pub fn tokenize(...) Error!... { return error.FeatureDisabled; }
};
```
Run `zig build check-parity` to verify mod/stub declaration parity.

## Step 2: Add the Feature Flag to build.zig

The build.zig is self-contained (no external build/ modules). Add the flag inline:

```zig
const feat_<name> = b.option(bool, "feat-<name>", "<Description>") orelse true;
```

Use `orelse true` for features enabled by default, `orelse false` for opt-in (like `feat_mobile`).

Then add it to the build options block where other flags are set:
```zig
options.addOption(bool, "feat_<name>", feat_<name>);
```

## Step 3: Add to Feature Catalog

Edit `src/core/feature_catalog.zig`. Three additions required:

1. Add the variant to the `Feature` enum.
2. Add the variant to the `ParitySpec` enum.
3. Add a catalog entry to the `all` array.

## Step 4: Add Conditional Import to root.zig

Edit `src/root.zig`. Add in the features section:

```zig
pub const <name> = if (build_options.feat_<name>) @import("features/<name>/mod.zig") else @import("features/<name>/stub.zig");
```

## Step 5: Format and Verify

```bash
zig build lib                      # Build passes
zig build lint                     # Format check
zig build check-parity             # Mod/stub parity
zig build test --summary all       # Tests pass
zig build doctor                   # Feature shows in config report
```

On Darwin 25+, the `check-parity` step compiles cleanly but may fail at runtime with `InvalidExe` — that's the known linker issue, not a code problem.

## Checklist

Use this list to confirm completeness before committing:

- [ ] `src/features/<name>/types.zig` created with shared error set
- [ ] `src/features/<name>/mod.zig` created with `isEnabled() -> true`
- [ ] `src/features/<name>/stub.zig` created with `isEnabled() -> false`, signatures match mod
- [ ] `build.zig` registers `feat_<name>` with `b.option` and `options.addOption`
- [ ] `src/core/feature_catalog.zig` `Feature` enum has `.<name>`
- [ ] `src/core/feature_catalog.zig` `ParitySpec` enum has `.<name>`
- [ ] `src/core/feature_catalog.zig` `all` array has catalog entry
- [ ] `src/root.zig` has conditional import line
- [ ] `zig build lib` passes
- [ ] `zig build lint` passes
- [ ] `zig build check-parity` compiles clean
- [ ] `./build.sh test --summary all` passes

## Common Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Stub signature differs from mod | Compile error when feature disabled | Copy mod.zig public API, replace bodies with no-ops |
| Missing from feature_catalog.zig | Comptime error in parity tests | Catalog entry is required before parity check passes |
| Using `@import("abi")` inside src/ | Circular import error | Use relative paths within the `abi` module |
| Sub-module in mod.zig not in stub | Compile error when disabled code references it | Add `pub const foo = struct {};` in stub |
| Missing `pub const types` re-export | `abi.<feature>.types` won't resolve | Always add `pub const types = @import("types.zig");` |
