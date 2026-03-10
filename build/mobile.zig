const std = @import("std");
const options_mod = @import("options.zig");
const modules = @import("modules.zig");

/// Register mobile cross-compilation targets (Android aarch64, iOS aarch64).
///
/// Only emits build artifacts when `feat_mobile` is set.
pub fn addMobileBuild(
    b: *std.Build,
    options: options_mod.BuildOptions,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const mobile_step = b.step("mobile", "Build for mobile targets (Android/iOS)");

    if (!options.feat_mobile) return mobile_step;

    // Android (aarch64)
    const android_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .linux,
        .abi = .android,
    });
    const android_build_opts = modules.createBuildOptionsModule(b, options);
    const android_shared_services = modules.createSharedServicesModule(b, android_build_opts, android_target, optimize);
    const android_core_module = modules.createCoreModule(b, android_target, optimize, android_build_opts);
    const abi_android = b.addLibrary(.{
        .name = "abi-android",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = android_target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    modules.wireAbiImports(abi_android.root_module, android_build_opts, android_shared_services, android_core_module);
    mobile_step.dependOn(&b.addInstallArtifact(abi_android, .{
        .dest_dir = .{ .override = .{ .custom = "mobile/android" } },
    }).step);

    // iOS (aarch64)
    const ios_target = b.resolveTargetQuery(.{
        .cpu_arch = .aarch64,
        .os_tag = .ios,
    });
    const ios_build_opts = modules.createBuildOptionsModule(b, options);
    const ios_shared_services = modules.createSharedServicesModule(b, ios_build_opts, ios_target, optimize);
    const ios_core_module = modules.createCoreModule(b, ios_target, optimize, ios_build_opts);
    const abi_ios = b.addLibrary(.{
        .name = "abi-ios",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = ios_target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    modules.wireAbiImports(abi_ios.root_module, ios_build_opts, ios_shared_services, ios_core_module);
    mobile_step.dependOn(&b.addInstallArtifact(abi_ios, .{
        .dest_dir = .{ .override = .{ .custom = "mobile/ios" } },
    }).step);

    return mobile_step;
}
