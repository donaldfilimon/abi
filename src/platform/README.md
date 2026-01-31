# Platform Module

Platform detection and abstraction layer for cross-platform code.

## Overview

The platform module provides runtime and compile-time detection of the operating system, CPU architecture, and hardware capabilities. It abstracts platform-specific differences and provides safe APIs for querying platform features across all supported targets (desktop, mobile, WebAssembly, and freestanding).

## Key Features

- **OS Detection**: Windows, Linux, macOS, BSD variants, iOS, WebAssembly, and freestanding
- **Architecture Detection**: x86-64, x86, ARM64, ARM, RISC-V, WASM32/64
- **SIMD Support**: Detects SIMD capability (SSE/AVX on x86, NEON on ARM, WASM SIMD)
- **Threading Safety**: Safe CPU count and threading APIs for WASM/freestanding targets
- **Platform Classification**: Queries for desktop, mobile, WebAssembly environments

## Core Types

### `Os` Enum
Operating system identifier with compile-time detection:
```zig
pub const Os = enum {
    windows, linux, macos, freebsd, netbsd, openbsd, dragonfly,
    ios, wasm, other
};
```

### `Arch` Enum
CPU architecture with SIMD capability detection:
```zig
pub const Arch = enum {
    x86_64, x86, aarch64, arm, riscv64, wasm32, wasm64, other
};
```
- `hasSimd()` returns true for x86, ARM, and WebAssembly (architecture-level SIMD support)

### `PlatformInfo` Struct
Runtime platform information:
```zig
pub const PlatformInfo = struct {
    os: Os,              // Operating system
    arch: Arch,          // CPU architecture
    max_threads: u32,    // Available CPU cores (1 on WASM/freestanding)
};
```

## Common APIs

| Function | Description |
|----------|-------------|
| `getPlatformInfo()` | Get runtime platform info (OS, architecture, thread count) |
| `supportsThreading()` | Check if threading is available (false for WASM/freestanding) |
| `getCpuCount()` | Get CPU count safely (returns 1 on unsupported targets) |
| `getDescription()` | Get human-readable platform string (e.g., "macOS Apple Silicon") |
| `hasSimd()` | Check if SIMD is available on current architecture |
| `isAppleSilicon()` | Check for Apple Silicon (macOS/iOS ARM64) |
| `isDesktop()` | Check if platform is a desktop OS |
| `isMobile()` | Check if platform is mobile (iOS) |
| `isWasm()` | Check if platform is WebAssembly |

## Usage Example

```zig
const abi = @import("abi");
const platform = abi.platform;
const std = @import("std");

pub fn main() !void {
    // Get platform information
    const info = platform.getPlatformInfo();
    std.debug.print("OS: {s}, Arch: {s}\n", .{
        @tagName(info.os),
        @tagName(info.arch),
    });

    // Check capabilities
    if (platform.hasSimd()) {
        std.debug.print("SIMD: available\n", .{});
    }

    if (platform.supportsThreading()) {
        std.debug.print("Threading: available ({d} cores)\n",
            .{platform.getCpuCount()});
    }

    // Platform-specific code
    if (platform.isAppleSilicon()) {
        // Use Accelerate framework (AMX support)
    }
}
```

## Sub-Modules

- **`detection.zig`**: OS/architecture mapping and compile-time detection
- **`cpu.zig`**: CPU feature fallback and capability detection

## WASM and Freestanding

On WebAssembly and freestanding targets:
- `supportsThreading()` returns `false`
- `getCpuCount()` returns `1` (safe for non-threaded code)
- `hasSimd()` correctly identifies WebAssembly SIMD support
- Threading APIs (`std.Thread`) are unavailable

This module ensures safe cross-compilation without runtime crashes on constrained environments.
