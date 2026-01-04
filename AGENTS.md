## ABI Framework Agentic Coding Guidelines

This file contains minimal, actionable guidance for developers working on the ABI framework.

### Build & Test Commands
```bash
zig build                     # Build all modules
zig build test --summary all  # Run full test suite
zig test <file>              # Run tests in a single file
zig build test --test-filter <name>  # Run tests matching pattern

# Feature flags
zig build -Denable-gpu=false      # Disable GPU features
zig build -Denable-ai=true       # Enable AI features
zig build -Doptimize=ReleaseFast  # Release build
```

### Code Style Guidelines

#### Documentation
- **File docs**: `//!` at top of each module, include usage examples when helpful
- **Public API**: `///` comments + `@param`/`@return` annotations for all exported functions
- **Inline docs**: Use `//` for brief explanations of complex logic

#### Naming Conventions
- **Types**: PascalCase (`HttpClient`, `ConfigError`, `TaskHandle`)
- **Functions/Variables**: snake_case (`getEnvVar`, `spawnTask`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_SIZE`, `DEFAULT_TIMEOUT`)
- **Error types**: PascalCase with descriptive names (`InvalidUrl`, `ReadFailed`)
- **Enum variants**: snake_case or PascalCase (be consistent per enum)

#### Imports & Modules
```zig
const std = @import("std");
const local_mod = @import("mod.zig");
const build_options = @import("build_options");

// NEVER use `usingnamespace`
```

#### Structs & Types
```zig
pub const MyStruct = struct {
    allocator: std.mem.Allocator,  // First field if allocator needed
    field: type = default_value,
    optional_field: ?type = null,

    pub fn init(allocator: std.mem.Allocator) MyStruct {
        return .{
            .allocator = allocator,
            .field = value,
        };
    }

    pub fn deinit(self: *MyStruct) void {
        // cleanup...
        self.* = undefined;
    }
};
```

#### Error Handling
- Use `try`/`?` for error propagation
- Define specific error types, avoid `anyerror` unless necessary
- Use error union `||` pattern to combine error sets:
```zig
pub const MyError = error{
    CustomError,
} || std.mem.Allocator.Error || std.Io.File.OpenError;
```

#### Memory Management
- Accept `std.mem.Allocator` as first argument for allocators
- Use `std.ArrayListUnmanaged` for struct fields (not `std.ArrayList`)
- Always pair allocations with `errdefer` cleanup
- Use `defer` for guaranteed cleanup

#### Collections
```zig
// For struct fields
var list: std.ArrayListUnmanaged(u8) = .empty;
try list.append(allocator, item);
defer list.deinit(allocator);

// For local variables
var list = std.ArrayList(u8).init(allocator);
defer list.deinit();
```

#### Conditional Compilation
```zig
if (build_options.enable_gpu) {
    // GPU-specific code
}

// Import disabled modules
const network_mod = if (build_options.enable_network)
    @import("network/mod.zig")
else
    @import("network/disabled.zig");
```

#### I/O Patterns (Zig 0.16)
```zig
// Use std.Io for all I/O
var io_backend = std.Io.Threaded.init(allocator, .{});
defer io_backend.deinit();
const io = io_backend.io();

// File operations
var file = try std.Io.Dir.cwd().createFile(io, path, .{});
defer file.close(io);

// Readers/writers
var buffer: [4096]u8 = undefined;
var reader = file.reader(io, &buffer);
var writer = file.writer(io, &buffer);
```

#### Testing
```zig
test "test name" {
    const allocator = std.testing.allocator;
    var thing = try Thing.init(allocator);
    defer thing.deinit();

    try std.testing.expectEqual(expected, actual);
    try std.testing.expectError(error.ExpectedErr, action());
}

// Feature-gated tests
if (!build_options.enable_gpu) return;
```

#### Formatting & Layout
- 4 spaces indentation (no tabs)
- Max 100 characters per line
- One blank line between functions
- Group related fields together
- Order: constants, types, functions, tests

### Feature Flags
| Flag | Default | Description |
|------|---------|-------------|
| `enable-gpu` | true | GPU acceleration |
| `enable-ai` | true | AI/ML features |
| `enable-web` | true | HTTP server/client |
| `enable-database` | true | Database features |
| `enable-network` | false | Distributed compute |
| `enable-profiling` | false | Profiling & metrics |

GPU backends: `gpu-cuda`, `gpu-vulkan`, `gpu-metal`, `gpu-webgpu`, `gpu-opengl`, `gpu-opengles`, `gpu-webgl2`

### Common Patterns

#### Type Parameters
```zig
pub fn MyHandle(comptime T: type) type {
    return struct {
        value: T,
        pub const Self = @This();
    };
}
```

#### Comptime Checks
```zig
comptime {
    if (!@hasField(@TypeOf(options), "required_field")) {
        @compileError("Missing required_field");
    }
}
```

#### Switch with Errors
```zig
const result = operation() catch |err| switch (err) {
    error.Temporary => continue,
    error.Fatal => return err,
};
```

### Environment Summary
```text
CWD: C:\Users\donald\abi
Approval policy: never
Sandbox: danger-full-access
Network: enabled
Shell: powershell
```

### Available Codex Skills
| Skill | Purpose |
|-------|---------|
| gh-address-comments | Auto-address GitHub PR comments |
| gh-fix-ci | Analyze and fix GitHub Actions failures |
| linear | Manage Linear tickets |
| notion-* | Various Notion integration skills |
| skill-creator | Create/modify skills |
| skill-installer | Install Codex skills |

*All skill files are located under `C:\Users\donald\.codex\skills\`.*
