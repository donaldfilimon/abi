pub const manifest = @import("manifest.zig");
pub const loader = @import("loader.zig");
pub const http_plugin = @import("http_plugin.zig");
pub const native_abi_v1 = @import("native_abi_v1.zig");
pub const native_plugin = @import("native_plugin.zig");

// ============================================================================
// Tests
// ============================================================================

test "module re-exports are accessible" {
    const std = @import("std");
    // Verify all submodule types are reachable
    _ = manifest.Manifest;
    _ = manifest.PluginEntry;
    _ = manifest.PluginKind;
    _ = loader.filterEnabledByKind;
    _ = native_abi_v1.ABI_VERSION;
    _ = native_abi_v1.PluginV1;
    _ = native_abi_v1.GenerateRequest;
    _ = native_abi_v1.GenerateResponse;
    _ = http_plugin.generate;
    _ = native_plugin.generate;
    _ = std;
}

test {
    // Pull in tests from all submodules
    _ = manifest;
    _ = loader;
    _ = http_plugin;
    _ = native_abi_v1;
    _ = native_plugin;
}
