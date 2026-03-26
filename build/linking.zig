const std = @import("std");

pub const DarwinRole = enum {
    static_lib,
    executable,
    test_artifact,
    parity_test,
};

pub fn linkDarwinArtifact(
    artifact: *std.Build.Step.Compile,
    role: DarwinRole,
    feat_gpu: bool,
    gpu_metal: bool,
) void {
    const b = artifact.step.owner;

    // macOS 26.4+ (Darwin 25.x): LLD cannot resolve system framework symbols
    // when using .tbd files that only list arm64e. Force Apple's native linker.
    // Defensive: linkIfDarwin already gates on macOS, but keep this guard in
    // case linkDarwinArtifact is called directly.
    if (artifact.root_module.resolved_target.?.result.os.tag == .macos) {
        artifact.use_lld = false;
    }

    // On macOS 26+ with a patched SDK overlay (--sysroot), add library and framework
    // search paths so zig's linker can find -lobjc, -framework IOKit, etc.
    // -L gets sysroot prepended by zig, so use relative /usr/lib.
    // -F does NOT get sysroot prepended, so use the absolute overlay path.
    if (b.sysroot) |sysroot| {
        artifact.root_module.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
        const fw_path = std.fmt.allocPrint(b.allocator, "{s}/System/Library/Frameworks", .{sysroot}) catch @panic("OOM");
        artifact.root_module.addFrameworkPath(.{ .cwd_relative = fw_path });
    }

    // Common libs for all roles except static_lib (which has its own set)
    if (role != .static_lib) {
        linkDarwinCommon(artifact, feat_gpu, true);
    }

    switch (role) {
        .static_lib => {
            for ([_][]const u8{ "System", "c" }) |lib| {
                artifact.root_module.linkSystemLibrary(lib, .{});
            }
            if (feat_gpu) {
                artifact.root_module.linkFramework("Accelerate", .{});
            }
            artifact.root_module.linkFramework("IOKit", .{});
            artifact.root_module.linkSystemLibrary("objc", .{});
            if (gpu_metal) {
                for ([_][]const u8{ "Metal", "MetalPerformanceShaders", "CoreGraphics" }) |framework| {
                    artifact.root_module.linkFramework(framework, .{});
                }
            }
        },
        .executable => {},
        .test_artifact => {
            if (gpu_metal) {
                for ([_][]const u8{ "Metal", "MetalPerformanceShaders" }) |framework| {
                    artifact.root_module.linkFramework(framework, .{});
                }
            }
        },
        .parity_test => {},
    }
}

/// Convenience wrapper: calls linkDarwinArtifact only when targeting macOS.
/// Use this instead of inline `if (target.os.tag == .macos)` checks.
pub fn linkIfDarwin(
    artifact: *std.Build.Step.Compile,
    role: DarwinRole,
    feat_gpu: bool,
    gpu_metal: bool,
) void {
    if (artifact.root_module.resolved_target) |rt| {
        if (rt.result.os.tag == .macos) {
            linkDarwinArtifact(artifact, role, feat_gpu, gpu_metal);
        }
    }
}

/// Shared Darwin framework set for executable, test_artifact, and parity_test roles.
fn linkDarwinCommon(artifact: *std.Build.Step.Compile, feat_gpu: bool, link_system: bool) void {
    if (link_system) {
        artifact.root_module.linkSystemLibrary("System", .{});
    }
    artifact.root_module.linkSystemLibrary("c", .{});
    artifact.root_module.linkSystemLibrary("objc", .{});
    artifact.root_module.linkFramework("IOKit", .{});
    artifact.root_module.linkFramework("CoreFoundation", .{});
    artifact.root_module.linkFramework("CoreGraphics", .{});
    if (feat_gpu) {
        artifact.root_module.linkFramework("Accelerate", .{});
    }
}
