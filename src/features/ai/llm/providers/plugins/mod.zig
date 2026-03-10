const std = @import("std");

pub const manifest = @import("manifest");
pub const loader = @import("loader");
pub const http_plugin = @import("http_plugin");
pub const native_abi_v1 = @import("native_abi_v1");
pub const native_plugin = @import("native_plugin");

// ============================================================================
// Tests
// ============================================================================

test "module re-exports are accessible" {
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

test {
    std.testing.refAllDecls(@This());
}
