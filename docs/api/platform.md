# platform

> Platform detection and abstraction.

**Source:** [`src/services/platform/mod.zig`](../../src/services/platform/mod.zig)

**Availability:** Always enabled

---

Platform Detection and Abstraction

Provides OS, architecture, and capability detection for cross-platform code.
This module consolidates all platform-specific detection and abstraction logic.

## Usage

```zig
const platform = @import("abi").platform;

const info = platform.getPlatformInfo();
std.debug.print("OS: {t}, Arch: {t}, Cores: {d}\n", .{
info.os,
info.arch,
info.max_threads,
});

if (platform.supportsThreading()) {
// Use multi-threaded code path
}
```

---

## API

### `pub fn getPlatformInfo() PlatformInfo`

<sup>**fn**</sup>

Get current platform information at runtime

### `pub fn supportsThreading() bool`

<sup>**fn**</sup>

Check if current platform supports threading
Returns false for freestanding and WASM targets

### `pub fn getCpuCount() usize`

<sup>**fn**</sup>

Get CPU count in a platform-safe manner
Returns 1 on freestanding/WASM targets where std.Thread is unavailable

### `pub fn getDescription() []const u8`

<sup>**fn**</sup>

Get a human-readable platform description string

### `pub fn hasSimd() bool`

<sup>**fn**</sup>

Check if SIMD is available on the current platform

### `pub fn isAppleSilicon() bool`

<sup>**fn**</sup>

Check if the current platform is Apple Silicon (macOS/iOS ARM64)

### `pub fn isDesktop() bool`

<sup>**fn**</sup>

Check if the current platform is a desktop OS

### `pub fn isMobile() bool`

<sup>**fn**</sup>

Check if the current platform is mobile

### `pub fn isWasm() bool`

<sup>**fn**</sup>

Check if the current platform is WebAssembly

---

*Generated automatically by `zig build gendocs`*
