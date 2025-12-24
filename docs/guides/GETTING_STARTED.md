# Getting Started

## Prerequisites
- Zig 0.16.x

## Build
```bash
zig build
zig build test
```

## Create a Minimal App
```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Use FrameworkConfiguration for full control, or FrameworkOptions for quick setup
    var framework = try abi.init(gpa.allocator(), abi.FrameworkConfiguration{});
    defer abi.shutdown(&framework);

    std.debug.print("ABI version: {s}\n", .{abi.version()});
}
```

## Next Steps
- Review docs/ARCHITECTURE.md
- Explore modules under `lib/features`
