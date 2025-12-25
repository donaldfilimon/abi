const std = @import("std");

const BuildOptions = struct {
    enable_gpu: bool,
    enable_ai: bool,
    enable_web: bool,
    enable_database: bool,
    enable_network: bool,
    enable_profiling: bool,
    gpu_cuda: bool,
    gpu_vulkan: bool,
    gpu_metal: bool,
    gpu_webgpu: bool,
    gpu_opengl: bool,
    gpu_opengles: bool,
    gpu_webgl2: bool,
};

fn readBuildOptions(b: *std.Build) BuildOptions {
    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU support") orelse true;
    const enable_ai = b.option(bool, "enable-ai", "Enable AI features") orelse true;
    const enable_web = b.option(bool, "enable-web", "Enable web features") orelse true;
    const enable_database =
        b.option(bool, "enable-database", "Enable database features") orelse true;
    const enable_network =
        b.option(bool, "enable-network", "Enable network distributed compute") orelse false;
    const enable_profiling =
        b.option(bool, "enable-profiling", "Enable profiling and metrics") orelse false;

    const gpu_cuda = b.option(bool, "gpu-cuda", "Enable CUDA GPU backend") orelse enable_gpu;
    const gpu_vulkan = b.option(bool, "gpu-vulkan", "Enable Vulkan GPU backend") orelse enable_gpu;
    const gpu_metal = b.option(bool, "gpu-metal", "Enable Metal GPU backend") orelse enable_gpu;
    const gpu_webgpu = b.option(bool, "gpu-webgpu", "Enable WebGPU backend") orelse enable_web;
    const gpu_opengl = b.option(bool, "gpu-opengl", "Enable OpenGL backend") orelse enable_gpu;
    const gpu_opengles =
        b.option(bool, "gpu-opengles", "Enable OpenGL ES backend") orelse enable_gpu;
    const gpu_webgl2 = b.option(bool, "gpu-webgl2", "Enable WebGL2 backend") orelse enable_web;

    return .{
        .enable_gpu = enable_gpu,
        .enable_ai = enable_ai,
        .enable_web = enable_web,
        .enable_database = enable_database,
        .enable_network = enable_network,
        .enable_profiling = enable_profiling,
        .gpu_cuda = gpu_cuda,
        .gpu_vulkan = gpu_vulkan,
        .gpu_metal = gpu_metal,
        .gpu_webgpu = gpu_webgpu,
        .gpu_opengl = gpu_opengl,
        .gpu_opengles = gpu_opengles,
        .gpu_webgl2 = gpu_webgl2,
    };
}

/// Create build options module with centralized defaults.
fn createBuildOptionsModule(b: *std.Build, options: BuildOptions) *std.Build.Module {
    var build_options = b.addOptions();

    // Package version
    build_options.addOption([]const u8, "package_version", "0.1.0");

    build_options.addOption(bool, "enable_gpu", options.enable_gpu);
    build_options.addOption(bool, "enable_ai", options.enable_ai);
    build_options.addOption(bool, "enable_web", options.enable_web);
    build_options.addOption(bool, "enable_database", options.enable_database);
    build_options.addOption(bool, "enable_network", options.enable_network);
    build_options.addOption(bool, "enable_profiling", options.enable_profiling);

    // GPU backend options
    build_options.addOption(bool, "gpu_cuda", options.gpu_cuda);
    build_options.addOption(bool, "gpu_vulkan", options.gpu_vulkan);
    build_options.addOption(bool, "gpu_metal", options.gpu_metal);
    build_options.addOption(bool, "gpu_webgpu", options.gpu_webgpu);
    build_options.addOption(bool, "gpu_opengl", options.gpu_opengl);
    build_options.addOption(bool, "gpu_opengles", options.gpu_opengles);
    build_options.addOption(bool, "gpu_webgl2", options.gpu_webgl2);

    return build_options.createModule();
}

fn pathExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create build options module
    const base_options = readBuildOptions(b);
    const build_options_module = createBuildOptionsModule(b, base_options);

    // Core library module
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_options_module);

    // CLI executable
    const cli_path: ?[]const u8 = if (pathExists("tools/cli/main.zig"))
        "tools/cli/main.zig"
    else if (pathExists("src/main.zig"))
        "src/main.zig"
    else
        null;
    if (cli_path) |path| {
        const exe = b.addExecutable(.{
            .name = "abi",
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
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
    } else {
        std.log.warn("CLI entrypoint not found; skipping CLI build", .{});
    }

    // Test suite
    const has_tests = pathExists("tests/mod.zig");
    if (has_tests) {
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
    } else {
        std.log.warn("tests/mod.zig not found; skipping test step", .{});
    }

    // Benchmark step
    const has_benchmark = pathExists("src/compute/runtime/benchmark.zig");
    if (has_benchmark) {
        const benchmark_exe = b.addExecutable(.{
            .name = "abi-benchmark",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/compute/runtime/benchmark_demo.zig"),
                .target = target,
                .optimize = .ReleaseFast,
            }),
        });
        benchmark_exe.root_module.addImport("abi", abi_module);

        const run_benchmark = b.addRunArtifact(benchmark_exe);

        const benchmark_step = b.step("benchmark", "Run performance benchmarks");
        benchmark_step.dependOn(&run_benchmark.step);
    } else {
        std.log.warn("src/compute/runtime/benchmark.zig not found; skipping benchmark step", .{});
    }

    // Performance profiling build
    if (cli_path) |path| {
        var profile_options = base_options;
        profile_options.enable_profiling = true;
        const profile_build_options_module = createBuildOptionsModule(b, profile_options);
        const abi_profile_module = b.addModule("abi-profile", .{
            .root_source_file = b.path("src/abi.zig"),
            .target = target,
            .optimize = optimize,
        });
        abi_profile_module.addImport("build_options", profile_build_options_module);

        const profile_exe = b.addExecutable(.{
            .name = "abi-profile",
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
                .target = target,
                .optimize = .ReleaseFast,
            }),
        });
        profile_exe.root_module.addImport("abi", abi_profile_module);
        b.installArtifact(profile_exe);

        const profile_step = b.step("profile", "Build with performance profiling");
        profile_step.dependOn(b.getInstallStep());
    }
}
