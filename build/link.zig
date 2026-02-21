const std = @import("std");

/// Apply macOS GPU framework links (Metal, CoreML, MPS, Foundation) to a
/// build module when targeting macOS with the Metal backend enabled.
///
/// Centralises the repeated linking block that previously appeared in
/// build.zig for the CLI exe, main tests, feature tests, and profile build.
pub fn applyFrameworkLinks(
    mod: *std.Build.Module,
    os_tag: std.Target.Os.Tag,
    gpu_metal: bool,
) void {
    if (os_tag == .macos and gpu_metal) {
        mod.linkFramework("Metal", .{});
        mod.linkFramework("CoreML", .{});
        mod.linkFramework("MetalPerformanceShaders", .{});
        mod.linkFramework("Foundation", .{});
    }
}
