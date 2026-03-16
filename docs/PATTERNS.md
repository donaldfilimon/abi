# ABI Codebase Patterns (Zig 0.16)

Key patterns and conventions used throughout the ABI codebase, specific to
Zig 0.16 (`0.16.0-dev.2905+5d71e3051`).

## Module System

### Single Module Ownership

All source files under `src/` belong to the single `abi` module. There are no
separate named modules for `core`, `shared_services`, or individual features.
The build system creates one `abi` module with `src/root.zig` as the root:

```zig
// build.zig
const abi_module = b.addModule("abi", .{
    .root_source_file = b.path("src/root.zig"),
    ...
});
```

### Import Conventions

- **Framework API**: `@import("abi")` — for code outside `src/` (examples, tools, tests)
- **Relative imports**: `@import("../core/config/mod.zig")` — within `src/` features
- **Build options**: `@import("build_options")` — comptime feature flags
- **Explicit `.zig` extensions**: required on all path imports (Zig 0.16 requirement)

```zig
// Correct (Zig 0.16)
const config = @import("../../core/config/mod.zig");

// Wrong — missing extension
const config = @import("../../core/config/mod");
```

## The mod/stub Contract

Every feature module (`src/features/<name>/`) follows a strict contract:

| File | Role |
|------|------|
| `mod.zig` | Real implementation (feature enabled) |
| `stub.zig` | API-compatible no-ops (feature disabled) |
| `types.zig` | Shared types imported by both mod and stub (when needed) |

`types.zig` is required when the module has shared public types that both
`mod.zig` and `stub.zig` must share. Thin modules without shared type
contracts may omit it.

### Rules

1. `stub.zig` must match `mod.zig` public signatures exactly
2. Shared types go in `types.zig` — both mod and stub import from it
3. Sub-module stubs are not required (only the top-level `stub.zig`)
4. When you change a public signature in `mod.zig`, update `stub.zig` immediately

### StubFeature Helpers

`src/core/stub_context.zig` provides helpers to reduce stub boilerplate:

```zig
// In stub.zig — common pattern
const stub_common = @import("../../services/shared/mod.zig").stub_common;
pub const Error = error{ FeatureDisabled } || stub_common.CommonError;
```

`StubFeature` and `StubFeatureNoConfig` handle the common case where a stub
needs to declare a feature struct with `init`, `deinit`, and status methods
that all return `error.FeatureDisabled` or default values.

## Feature Flag Gating

Features are toggled via build options defined in `build/options.zig`:

```zig
// Reading flags at comptime
const build_options = @import("build_options");
const feat_gpu = build_options.feat_gpu;

// Conditional compilation
if (feat_gpu) {
    // real GPU code
} else {
    // stub path
}
```

25 feature flags exist, all enabled by default. Disable with
`-Dfeat-<name>=false`. The 54 validated flag combinations live in
`build/flags.zig`.

## Zig 0.16 API Patterns

### Time

```zig
// 0.16 — use unixSeconds()
const now = std.time.unixSeconds();

// Not: std.time.timestamp()
```

### File I/O

```zig
// 0.16 streaming write
file.writeStreamingAll(io, data);

// Not: file.writeAll(data)
```

### Directory Creation

```zig
// 0.16
std.Io.Dir.createDirPath(.cwd(), io, path);

// Not: std.fs.makeDirAbsolute(path)
```

### Build System Paths

```zig
// 0.16 — use b.path() which returns LazyPath with .cwd_relative
b.path("src/root.zig")

// .root_module instead of .root_source_file on executables
const exe = b.addExecutable(.{
    .name = "app",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        ...
    }),
});
```

### Collections

```zig
// ArrayListUnmanaged initialization
var list: std.ArrayListUnmanaged(T) = .empty;

// Not: .{} or .init()

// HashMap iteration
var it = map.valueIterator();
while (it.next()) |val| { ... }

// Not: map.values()
```

### Enum Conversion

```zig
// 0.16
const e = @enumFromInt(x);

// Not: @intToEnum(EnumType, x)
```

## Darwin 25+ Workarounds

On macOS 26+ (Darwin Tahoe), the Zig 0.16 linker cannot resolve system
libraries. The codebase handles this with:

1. **`is_blocked_darwin`** constant in `build.zig` — gates linking behavior
2. **`use_llvm = true`** on all artifacts — allows compilation without linking
3. **`./tools/scripts/run_build.sh`** — wrapper script for build commands
4. **Compile-only steps** — `test` becomes typecheck, `gendocs` becomes compile-check

```zig
// build.zig pattern
const is_blocked_darwin = builtin.os.tag == .macos and
    builtin.os.version_range.semver.min.major >= 26;

if (is_blocked_darwin) {
    exe.use_llvm = true;
    // Don't install or run — just compile
} else {
    b.installArtifact(exe);
}
```

Format checks always work regardless of platform:
```bash
zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/
```

## Error Handling

- Use explicit error sets, not `anyerror`
- Propagate errors with `try`, never silently swallow
- Feature stubs return `error.FeatureDisabled` for operations that require
  the feature to be enabled

```zig
// Explicit error set
pub const InitError = error{
    OutOfMemory,
    InvalidConfig,
    FeatureDisabled,
};

// Propagation
pub fn init(alloc: Allocator) InitError!Self {
    const config = try loadConfig(alloc);
    ...
}
```

## Naming Conventions

| Kind | Convention | Example |
|------|-----------|---------|
| Files | `lower_snake_case` | `feature_catalog.zig` |
| Functions | `lower_snake_case` | `initDefault()` |
| Types | `PascalCase` | `BuildOptions` |
| Error sets | `PascalCase` | `InitError` |
| Constants | `lower_snake_case` | `max_retries` |
| Comptime params | `PascalCase` | `comptime T: type` |

## Formatting

- Always use `zig fmt` — never manual alignment
- Never run `zig fmt .` from the repo root (walks vendored fixtures)
- Target specific paths: `zig fmt --check build.zig build/ src/ tools/ examples/ tests/ bindings/ lang/`
