const std = @import("std");
const options_mod = @import("options.zig");
const modules = @import("modules.zig");

/// Register mobile cross-compilation targets (Android aarch64, iOS aarch64).
///
/// Only emits build artifacts when `enable_mobile` is set.
pub fn addMobileBuild(
    b: *std.Build,
    options: options_mod.BuildOptions,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const mobile_step = b.step("mobile", "Build for mobile targets (Android/iOS)");

    if (!options.enable_mobile) return mobile_step;

    // Android (aarch64)
    const android_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .android,
    });
    const abi_android = b.addLibrary(.{
        .name = "abi-android",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/abi.zig"),
            .target = android_target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    abi_android.root_module.addImport("build_options", modules.createBuildOptionsModule(b, options));
    mobile_step.dependOn(&b.addInstallArtifact(abi_android, .{
        .dest_dir = .{ .override = .{ .custom = "mobile/android" } },
    }).step);

    // iOS (aarch64)
    const ios_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .ios,
    });
    const abi_ios = b.addLibrary(.{
        .name = "abi-ios",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/abi.zig"),
            .target = ios_target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    abi_ios.root_module.addImport("build_options", modules.createBuildOptionsModule(b, options));
    mobile_step.dependOn(&b.addInstallArtifact(abi_ios, .{
        .dest_dir = .{ .override = .{ .custom = "mobile/ios" } },
    }).step);

    return mobile_step;
}
