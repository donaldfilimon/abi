# platform

> Platform Detection and Abstraction

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

**Source:** [`src/services/platform/mod.zig`](../../src/services/platform/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-fn-getplatforminfo-platforminfo"></a>`pub fn getPlatformInfo() PlatformInfo`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L38)

Get current platform information at runtime

### <a id="pub-fn-supportsthreading-bool"></a>`pub fn supportsThreading() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L44)

Check if current platform supports threading
Returns false for freestanding and WASM targets

### <a id="pub-fn-getcpucount-usize"></a>`pub fn getCpuCount() usize`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L50)

Get CPU count in a platform-safe manner
Returns 1 on freestanding/WASM targets where std.Thread is unavailable

### <a id="pub-fn-getdescription-const-u8"></a>`pub fn getDescription() []const u8`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L55)

Get a human-readable platform description string

### <a id="pub-fn-hassimd-bool"></a>`pub fn hasSimd() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L92)

Check if SIMD is available on the current platform

### <a id="pub-fn-isapplesilicon-bool"></a>`pub fn isAppleSilicon() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L97)

Check if the current platform is Apple Silicon (macOS/iOS ARM64)

### <a id="pub-fn-isdesktop-bool"></a>`pub fn isDesktop() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L104)

Check if the current platform is a desktop OS

### <a id="pub-fn-ismobile-bool"></a>`pub fn isMobile() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L112)

Check if the current platform is mobile

### <a id="pub-fn-isbsd-bool"></a>`pub fn isBsd() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L121)

Check if the current platform is a BSD variant

### <a id="pub-fn-isposix-bool"></a>`pub fn isPosix() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L126)

Check if the current platform is POSIX-compliant

### <a id="pub-fn-isapple-bool"></a>`pub fn isApple() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L131)

Check if the current platform is an Apple OS

### <a id="pub-fn-iswasm-bool"></a>`pub fn isWasm() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L136)

Check if the current platform is WebAssembly



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use the `$zig-master` Codex skill for ABI Zig validation, docs generation, and build-wiring changes.
