const std = @import("std");

/// Build configuration and feature flags
const BuildConfig = struct {
    // Feature toggles
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_database: bool = true,
    enable_web: bool = true,
    enable_monitoring: bool = true,
    
    // GPU backend toggles
    gpu_cuda: bool = false,
    gpu_vulkan: bool = false,
    gpu_metal: bool = false,
    gpu_webgpu: bool = false,
    
    // Build metadata
    package_version: []const u8 = "0.2.0",
    
    fn fromBuilder(b: *std.Build) BuildConfig {
        return .{
            .enable_ai = b.option(bool, "enable-ai", "Enable AI features") orelse true,
            .enable_gpu = b.option(bool, "enable-gpu", "Enable GPU acceleration") orelse true,
            .enable_database = b.option(bool, "enable-database", "Enable database features") orelse true,
            .enable_web = b.option(bool, "enable-web", "Enable web server") orelse true,
            .enable_monitoring = b.option(bool, "enable-monitoring", "Enable monitoring") orelse true,
            .gpu_cuda = b.option(bool, "gpu-cuda", "Enable CUDA support") orelse false,
            .gpu_vulkan = b.option(bool, "gpu-vulkan", "Enable Vulkan support") orelse false,
            .gpu_metal = b.option(bool, "gpu-metal", "Enable Metal support") orelse false,
            .gpu_webgpu = b.option(bool, "gpu-webgpu", "Enable WebGPU support") orelse false,
            .package_version = b.option([]const u8, "version", "Package version") orelse "0.2.0",
        };
    }
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const config = BuildConfig.fromBuilder(b);
    
    // Build options for compile-time configuration
    const build_options = createBuildOptions(b, config);
    
    // Core library module
    const abi_module = createAbiModule(b, target, optimize, build_options);
    
    // CLI executable
    const cli_exe = buildCLI(b, target, optimize, abi_module);
    b.installArtifact(cli_exe);

    // Run step
    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&b.addRunArtifact(cli_exe).step);

    // Core library module
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("lib/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    // CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = abi_module,
    });
    exe.addSourceFile(.{ .path = "tools/cli/main.zig" });
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
        .root_module = abi_module,
    });
    main_tests.addSourceFile(.{ .path = "tests/mod.zig" });

    const run_main_tests = b.addRunArtifact(main_tests);
    run_main_tests.skip_foreign_checks = true;

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_main_tests.step);
}

fn buildTools(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
) void {
    // Performance profiler
    const profiler_exe = b.addExecutable(.{
        .name = "performance_profiler",
        .root_source_file = b.path("tools/performance_profiler.zig"),
        .target = target,
        .optimize = optimize,
    });
    profiler_exe.root_module.addImport("abi", abi_module);
    
    const install_profiler = b.addInstallArtifact(profiler_exe, .{
        .dest_dir = .{ .override = .{ .custom = "tools" } },
    });
    
    const tools_step = b.step("tools", "Build development tools");
    tools_step.dependOn(&install_profiler.step);
}