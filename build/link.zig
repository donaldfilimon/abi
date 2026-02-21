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

const required_metal_framework_paths = [_][]const u8{
    "/System/Library/Frameworks/Metal.framework",
    "/System/Library/Frameworks/CoreML.framework",
    "/System/Library/Frameworks/MetalPerformanceShaders.framework",
    "/System/Library/Frameworks/Foundation.framework",
};

pub fn canLinkMetalFrameworks(io: std.Io, os_tag: std.Target.Os.Tag) bool {
    if (os_tag != .macos) return false;

    // Prefer SDK probe first. This avoids false negatives from low-level
    // path checks in some Zig 0.16 host I/O setups.
    if (commandSucceeds(io, &.{ "/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path" })) {
        return true;
    }

    for (required_metal_framework_paths) |path| {
        if (!commandSucceeds(io, &.{ "/usr/bin/test", "-e", path })) {
            return false;
        }
    }
    return true;
}

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
        "explicit gpu-backend=metal requested for macOS target, but required Apple frameworks are unavailable. " ++ "Install an Apple SDK/Xcode Command Line Tools or use -Dgpu-backend=auto/vulkan.",
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
