const std = @import("std");

/// Link macOS GPU frameworks (Metal, CoreML, MPS, Foundation) into a module
/// when targeting macOS with the Metal backend enabled.
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

const required_metal_framework_paths = [_][]const u8{
    "/System/Library/Frameworks/Metal.framework",
    "/System/Library/Frameworks/CoreML.framework",
    "/System/Library/Frameworks/MetalPerformanceShaders.framework",
    "/System/Library/Frameworks/Foundation.framework",
};

/// Probe whether Metal frameworks can be linked on this host.  Tries
/// `xcrun --sdk macosx --show-sdk-path` first; falls back to checking
/// framework paths directly.
pub fn canLinkMetalFrameworks(io: std.Io, os_tag: std.Target.Os.Tag) bool {
    if (os_tag != .macos) return false;

    if (commandSucceeds(io, &.{ "/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path" }))
        return true;

    for (required_metal_framework_paths) |path|
        if (!commandSucceeds(io, &.{ "/usr/bin/test", "-e", path })) return false;
    return true;
}

/// Abort the build when the user explicitly requested `-Dgpu-backend=metal`
/// but the required frameworks are not available.
pub fn validateMetalBackendRequest(
    b: *std.Build,
    backend_arg: ?[]const u8,
    os_tag: std.Target.Os.Tag,
    can_link_metal: bool,
) void {
    _ = b;
    if (os_tag != .macos) return;
    if (!isExplicitMetalRequested(backend_arg)) return;
    if (can_link_metal) return;

    std.debug.panic(
        "explicit gpu-backend=metal requested but Apple frameworks are unavailable. " ++
            "Install Xcode Command Line Tools or use -Dgpu-backend=auto/vulkan.",
        .{},
    );
}

fn isExplicitMetalRequested(backend_arg: ?[]const u8) bool {
    const arg = backend_arg orelse return false;
    var it = std.mem.splitScalar(u8, arg, ',');
    while (it.next()) |raw| {
        const token = std.mem.trim(u8, raw, " \t");
        if (token.len == 0) continue;
        if (std.ascii.eqlIgnoreCase(token, "metal")) return true;
    }
    return false;
}

fn commandSucceeds(io: std.Io, argv: []const []const u8) bool {
    var child = std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    }) catch return false;

    const term = child.wait(io) catch return false;
    return switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
}
