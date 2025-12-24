const std = @import("std");

/// Create build options module with centralized defaults
fn createBuildOptions(b: *std.Build) *std.Build.Module {
    const build_options = b.addOptions();

    // Package version
    build_options.addOption([]const u8, "package_version", "0.1.0");

    // Feature flags - all default to true for full functionality
    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU support") orelse true;
    const enable_ai = b.option(bool, "enable-ai", "Enable AI features") orelse true;
    const enable_web = b.option(bool, "enable-web", "Enable web features") orelse true;
    const enable_database = b.option(bool, "enable-database", "Enable database features") orelse true;

    // GPU backend selection
    const gpu_cuda = b.option(bool, "gpu-cuda", "Enable CUDA GPU backend") orelse enable_gpu;
    const gpu_vulkan = b.option(bool, "gpu-vulkan", "Enable Vulkan GPU backend") orelse enable_gpu;
    const gpu_metal = b.option(bool, "gpu-metal", "Enable Metal GPU backend") orelse enable_gpu;
    const gpu_webgpu = b.option(bool, "gpu-webgpu", "Enable WebGPU backend") orelse enable_web;

    build_options.addOption(bool, "enable_gpu", enable_gpu);
    build_options.addOption(bool, "enable_ai", enable_ai);
    build_options.addOption(bool, "enable_web", enable_web);
    build_options.addOption(bool, "enable_database", enable_database);

    // GPU backend options
    build_options.addOption(bool, "gpu_cuda", gpu_cuda);
    build_options.addOption(bool, "gpu_vulkan", gpu_vulkan);
    build_options.addOption(bool, "gpu_metal", gpu_metal);
    build_options.addOption(bool, "gpu_webgpu", gpu_webgpu);

    return build_options.createModule();
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create build options module
    const build_options_module = createBuildOptions(b);

    // Core library module
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("lib/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_options_module);

    // CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("abi", abi_module);
    b.installArtifact(exe);

    // Run step for CLI
    const run_cli = b.addRunArtifact(exe);
    run_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cli.addArgs(args);
    }

    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&run_cli.step);

    // Test suite
    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    main_tests.root_module.addImport("abi", abi_module);

    const run_main_tests = b.addRunArtifact(main_tests);
    run_main_tests.skip_foreign_checks = true;

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_main_tests.step);

    // Performance profiling build
    const profile_exe = b.addExecutable(.{
        .name = "abi-profile",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    profile_exe.root_module.addImport("abi", abi_module);

    const profile_step = b.step("profile", "Build with performance profiling");
    profile_step.dependOn(b.getInstallStep());

    // Note: Add documentation generation step
}
