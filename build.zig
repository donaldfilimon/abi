const std = @import("std");

/// Build configuration and feature flags
const BuildConfig = struct {
    // Feature toggles
    enable_ai: bool = true,
    enable_gpu: bool = true,
    enable_database: bool = true,
    enable_web: bool = true,
    enable_monitoring: bool = true,
    enable_connectors: bool = true,

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
            .enable_connectors = b.option(bool, "enable-connectors", "Enable connectors") orelse true,
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
    const cli_exe = buildCLI(b, target, optimize, abi_module, build_options);
    b.installArtifact(cli_exe);

    // Run step for CLI
    const run_cli = b.addRunArtifact(cli_exe);
    run_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cli.addArgs(args);

    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&run_cli.step);

    // Test suites
    buildTests(b, target, optimize, abi_module, build_options);

    // Examples
    buildExamples(b, target, optimize, abi_module, build_options);

    // Benchmarks
    buildBenchmarks(b, target, optimize, abi_module, build_options);

    // Documentation
    buildDocs(b, target, optimize, abi_module, build_options);

    // Additional tools
    buildTools(b, target, optimize, abi_module, build_options);
}

fn createBuildOptions(b: *std.Build, config: BuildConfig) *std.Build.Step.Options {
    const options = b.addOptions();

    // Package metadata
    options.addOption([]const u8, "package_version", config.package_version);
    options.addOption([]const u8, "package_name", "abi");

    // Feature flags
    options.addOption(bool, "enable_ai", config.enable_ai);
    options.addOption(bool, "enable_gpu", config.enable_gpu);
    options.addOption(bool, "enable_database", config.enable_database);
    options.addOption(bool, "enable_web", config.enable_web);
    options.addOption(bool, "enable_monitoring", config.enable_monitoring);
    options.addOption(bool, "enable_connectors", config.enable_connectors);

    // GPU backend flags
    options.addOption(bool, "gpu_cuda", config.gpu_cuda);
    options.addOption(bool, "gpu_vulkan", config.gpu_vulkan);
    options.addOption(bool, "gpu_metal", config.gpu_metal);
    options.addOption(bool, "gpu_webgpu", config.gpu_webgpu);

    return options;
}

fn createAbiModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Step.Options,
) *std.Build.Module {
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    abi_mod.addOptions("build_options", build_options);
    return abi_mod;
}

fn buildCLI(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = b.path("src/comprehensive_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.strip = optimize != .Debug; // reduce binary size by default in non-Debug

    exe.root_module.addImport("abi", abi_module);
    exe.root_module.addOptions("build_options", build_options);

    return exe;
}

fn buildTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
) void {
    // Main test suite
    const main_tests = b.addTest(.{
        .name = "abi_tests",
        .root_source_file = b.path("src/tests/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_tests.root_module.addImport("abi", abi_module);
    main_tests.root_module.addOptions("build_options", build_options);

    const run_main_tests = b.addRunArtifact(main_tests);
    run_main_tests.skip_foreign_checks = true;

    const unit_test_step = b.step("test", "Run unit tests");
    unit_test_step.dependOn(&run_main_tests.step);

    // Integration tests
    const integration_tests = b.addTest(.{
        .name = "integration_tests",
        .root_source_file = b.path("tests/integration/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_tests.root_module.addImport("abi", abi_module);
    integration_tests.root_module.addOptions("build_options", build_options);

    const run_integration_tests = b.addRunArtifact(integration_tests);
    run_integration_tests.skip_foreign_checks = true;

    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // All tests aggregator
    const all_test_step = b.step("test-all", "Run all tests");
    all_test_step.dependOn(&run_main_tests.step);
    all_test_step.dependOn(&run_integration_tests.step);
}

fn buildExamples(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
) void {
    const examples = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "ai_demo", .path = "src/examples/ai_demo.zig" },
        .{ .name = "agent_demo", .path = "src/examples/agent_subsystem_demo.zig" },
        .{ .name = "gpu_demo", .path = "src/examples/gpu_acceleration_demo.zig" },
        .{ .name = "transformer", .path = "src/examples/transformer_complete_example.zig" },
        .{ .name = "rl_example", .path = "src/examples/rl_complete_example.zig" },
    };

    for (examples) |example| {
        const exe = b.addExecutable(.{
            .name = example.name,
            .root_source_file = b.path(example.path),
            .target = target,
            .optimize = optimize,
        });
        exe.strip = optimize != .Debug;
        exe.root_module.addImport("abi", abi_module);
        exe.root_module.addOptions("build_options", build_options);

        const install_exe = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "examples" } } });

        const build_step = b.step(b.fmt("example-{s}", .{example.name}), b.fmt("Build {s} example", .{example.name}));
        build_step.dependOn(&install_exe.step);

        const run_example = b.addRunArtifact(exe);
        const run_step = b.step(b.fmt("run-{s}", .{example.name}), b.fmt("Run {s} example", .{example.name}));
        run_step.dependOn(&run_example.step);
    }

    const all_examples_step = b.step("examples", "Build all examples");
    for (examples) |example| {
        const step_name = b.fmt("example-{s}", .{example.name});
        if (b.top_level_steps.get(step_name)) |step| all_examples_step.dependOn(&step.step);
    }
}

fn buildBenchmarks(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
) void {
    const bench_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_exe.strip = optimize != .Debug;
    bench_exe.root_module.addImport("abi", abi_module);
    bench_exe.root_module.addOptions("build_options", build_options);

    const install_bench = b.addInstallArtifact(bench_exe, .{ .dest_dir = .{ .override = .{ .custom = "bench" } } });

    const bench_step = b.step("bench", "Build benchmarks");
    bench_step.dependOn(&install_bench.step);

    const run_bench = b.addRunArtifact(bench_exe);
    const run_bench_step = b.step("run-bench", "Run benchmarks");
    run_bench_step.dependOn(&run_bench.step);
}

fn buildDocs(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
) void {
    const docs_exe = b.addExecutable(.{
        .name = "docs_generator",
        .root_source_file = b.path("src/tools/docs_generator.zig"),
        .target = target,
        .optimize = optimize,
    });
    docs_exe.strip = optimize != .Debug;
    docs_exe.root_module.addImport("abi", abi_module);
    docs_exe.root_module.addOptions("build_options", build_options);

    const run_docs = b.addRunArtifact(docs_exe);
    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&run_docs.step);

    // Built-in documentation for the abi library
    const lib_step = b.addSharedLibrary(.{
        .name = "abi",
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_step.strip = optimize != .Debug;
    lib_step.root_module.addOptions("build_options", build_options);

    const install_docs = b.addInstallDirectory(.{
        .source_dir = lib_step.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs/api",
    });

    const docs_auto_step = b.step("docs-auto", "Generate Zig autodocs");
    docs_auto_step.dependOn(&install_docs.step);
}

fn buildTools(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
) void {
    // Performance profiler
    const profiler_exe = b.addExecutable(.{
        .name = "performance_profiler",
        .root_source_file = b.path("tools/performance_profiler.zig"),
        .target = target,
        .optimize = optimize,
    });
    profiler_exe.strip = optimize != .Debug;
    profiler_exe.root_module.addImport("abi", abi_module);
    profiler_exe.root_module.addOptions("build_options", build_options);

    const install_profiler = b.addInstallArtifact(profiler_exe, .{ .dest_dir = .{ .override = .{ .custom = "tools" } } });

    const tools_step = b.step("tools", "Build development tools");
    tools_step.dependOn(&install_profiler.step);
}
