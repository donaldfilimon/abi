# platform

> Platform detection and abstraction.

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

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L89)

Check if SIMD is available on the current platform

### <a id="pub-fn-isapplesilicon-bool"></a>`pub fn isAppleSilicon() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L94)

Check if the current platform is Apple Silicon (macOS/iOS ARM64)

### <a id="pub-fn-isdesktop-bool"></a>`pub fn isDesktop() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L101)

Check if the current platform is a desktop OS

### <a id="pub-fn-ismobile-bool"></a>`pub fn isMobile() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L109)

Check if the current platform is mobile

### <a id="pub-fn-iswasm-bool"></a>`pub fn isWasm() bool`

<sup>**fn**</sup> | [source](../../src/services/platform/mod.zig#L115)

Check if the current platform is WebAssembly

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
