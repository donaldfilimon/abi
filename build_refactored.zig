const std = @import("std");

/// Enhanced build configuration with better modularity
const BuildConfig = struct {
    // Feature toggles with dependencies
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
    gpu_opencl: bool = false,

    // Development features
    enable_testing: bool = true,
    enable_benchmarks: bool = true,
    enable_examples: bool = true,
    enable_docs: bool = true,

    // Performance options
    enable_simd: bool = true,
    enable_lto: bool = false,
    strip_debug: bool = false,

    // Build metadata
    package_version: []const u8 = "0.2.0",
    build_timestamp: []const u8 = "",
    git_commit: []const u8 = "",

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
            .gpu_opencl = b.option(bool, "gpu-opencl", "Enable OpenCL support") orelse false,
            
            .enable_testing = b.option(bool, "enable-testing", "Enable test suite") orelse true,
            .enable_benchmarks = b.option(bool, "enable-benchmarks", "Enable benchmarks") orelse true,
            .enable_examples = b.option(bool, "enable-examples", "Enable examples") orelse true,
            .enable_docs = b.option(bool, "enable-docs", "Enable documentation") orelse true,
            
            .enable_simd = b.option(bool, "enable-simd", "Enable SIMD optimizations") orelse true,
            .enable_lto = b.option(bool, "enable-lto", "Enable link-time optimization") orelse false,
            .strip_debug = b.option(bool, "strip-debug", "Strip debug information") orelse false,
            
            .package_version = b.option([]const u8, "version", "Package version") orelse "0.2.0",
            .build_timestamp = getBuildTimestamp(b),
            .git_commit = getGitCommit(b),
        };
    }

    fn validate(self: BuildConfig) !void {
        // Validate feature dependencies
        if (self.enable_ai and !self.enable_database) {
            std.log.warn("AI features work best with database enabled", .{});
        }
        
        if (self.enable_gpu and !(self.gpu_cuda or self.gpu_vulkan or self.gpu_metal or self.gpu_webgpu or self.gpu_opencl)) {
            std.log.warn("GPU enabled but no backends selected - using CPU fallback", .{});
        }
        
        if (self.enable_lto and self.strip_debug) {
            std.log.info("LTO and debug stripping enabled - optimizing for size", .{});
        }
    }
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const config = BuildConfig.fromBuilder(b);
    
    // Validate configuration
    config.validate() catch |err| {
        std.log.err("Build configuration validation failed: {s}", .{@errorName(err)});
        return;
    };

    // Build options for compile-time configuration
    const build_options = createBuildOptions(b, config);

    // Core library module with conditional features
    const abi_module = createAbiModule(b, target, optimize, build_options, config);

    // CLI executable
    if (config.enable_web or config.enable_ai or config.enable_database) {
        const cli_exe = buildCLI(b, target, optimize, abi_module, config);
        b.installArtifact(cli_exe);
        
        // Run step for CLI
        const run_cli = b.addRunArtifact(cli_exe);
        run_cli.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_cli.addArgs(args);
        }

        const run_step = b.step("run", "Run the ABI CLI");
        run_step.dependOn(&run_cli.step);
    }

    // Test suite
    if (config.enable_testing) {
        buildTests(b, target, optimize, abi_module, build_options, config);
    }

    // Examples
    if (config.enable_examples) {
        buildExamples(b, target, optimize, abi_module, config);
    }

    // Benchmarks
    if (config.enable_benchmarks) {
        buildBenchmarks(b, target, optimize, abi_module, config);
    }

    // Documentation
    if (config.enable_docs) {
        buildDocs(b, target, optimize, abi_module);
    }

    // Additional tools
    buildTools(b, target, optimize, abi_module, config);
    
    // Custom build steps
    addCustomSteps(b, config);
}

fn createBuildOptions(b: *std.Build, config: BuildConfig) *std.Build.Step.Options {
    const options = b.addOptions();

    // Package metadata
    options.addOption([]const u8, "package_version", config.package_version);
    options.addOption([]const u8, "package_name", "abi");
    options.addOption([]const u8, "build_timestamp", config.build_timestamp);
    options.addOption([]const u8, "git_commit", config.git_commit);

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
    options.addOption(bool, "gpu_opencl", config.gpu_opencl);

    // Performance flags
    options.addOption(bool, "enable_simd", config.enable_simd);

    return options;
}

fn createAbiModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Step.Options,
    config: BuildConfig,
) *std.Build.Module {
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    abi_mod.addOptions("build_options", build_options);
    
    // Add conditional compilation flags
    if (config.enable_simd) {
        abi_mod.addCMacro("ABI_ENABLE_SIMD", "1");
    }
    
    // Platform-specific optimizations
    switch (target.result.os.tag) {
        .windows => {
            if (config.gpu_cuda) {
                // Add CUDA library paths if available
                abi_mod.addLibraryPath(b.path("deps/cuda/lib"));
            }
        },
        .linux => {
            if (config.gpu_vulkan) {
                abi_mod.linkSystemLibrary("vulkan", .{});
            }
        },
        .macos => {
            if (config.gpu_metal) {
                abi_mod.linkFramework("Metal");
                abi_mod.linkFramework("MetalKit");
            }
        },
        else => {},
    }

    return abi_mod;
}

fn buildCLI(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    config: BuildConfig,
) *std.Build.Step.Compile {
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = b.path("src/comprehensive_cli.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("abi", abi_module);
    
    // Performance optimizations
    exe.link_function_sections = true;
    exe.link_data_sections = true;
    
    if (optimize != .Debug) {
        if (config.strip_debug) {
            exe.strip = true;
        }
        exe.link_gc_sections = true;
    }
    
    if (config.enable_lto and optimize == .ReleaseFast) {
        exe.want_lto = true;
    }

    return exe;
}

fn buildTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
    config: BuildConfig,
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

    // Unit tests step
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

    // Feature-specific tests
    if (config.enable_ai) {
        buildFeatureTests(b, target, optimize, abi_module, build_options, "ai");
    }
    
    if (config.enable_gpu) {
        buildFeatureTests(b, target, optimize, abi_module, build_options, "gpu");
    }
    
    if (config.enable_database) {
        buildFeatureTests(b, target, optimize, abi_module, build_options, "database");
    }

    // All tests step
    const all_test_step = b.step("test-all", "Run all tests");
    all_test_step.dependOn(&run_main_tests.step);
    all_test_step.dependOn(&run_integration_tests.step);
}

fn buildFeatureTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    build_options: *std.Build.Step.Options,
    feature_name: []const u8,
) void {
    const test_path = b.fmt("src/features/{s}/tests.zig", .{feature_name});
    
    // Check if feature test file exists
    std.fs.cwd().access(test_path, .{}) catch return;
    
    const feature_tests = b.addTest(.{
        .name = b.fmt("{s}_tests", .{feature_name}),
        .root_source_file = b.path(test_path),
        .target = target,
        .optimize = optimize,
    });
    feature_tests.root_module.addImport("abi", abi_module);
    feature_tests.root_module.addOptions("build_options", build_options);

    const run_feature_tests = b.addRunArtifact(feature_tests);
    const feature_test_step = b.step(
        b.fmt("test-{s}", .{feature_name}),
        b.fmt("Run {s} feature tests", .{feature_name}),
    );
    feature_test_step.dependOn(&run_feature_tests.step);
}

fn buildExamples(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    config: BuildConfig,
) void {
    const examples = [_]struct { name: []const u8, path: []const u8, requires: []const []const u8 }{
        .{ .name = "ai_demo", .path = "src/examples/ai_demo.zig", .requires = &.{"ai"} },
        .{ .name = "agent_demo", .path = "src/examples/agent_subsystem_demo.zig", .requires = &.{"ai"} },
        .{ .name = "gpu_demo", .path = "src/examples/gpu_acceleration_demo.zig", .requires = &.{"gpu"} },
        .{ .name = "transformer", .path = "src/examples/transformer_complete_example.zig", .requires = &.{ "ai", "gpu" } },
        .{ .name = "rl_example", .path = "src/examples/rl_complete_example.zig", .requires = &.{"ai"} },
        .{ .name = "web_server", .path = "src/examples/web_server_demo.zig", .requires = &.{"web"} },
        .{ .name = "monitoring", .path = "src/examples/monitoring.zig", .requires = &.{"monitoring"} },
    };

    for (examples) |example| {
        // Check if required features are enabled
        var can_build = true;
        for (example.requires) |required_feature| {
            const feature_enabled = if (std.mem.eql(u8, required_feature, "ai")) config.enable_ai else if (std.mem.eql(u8, required_feature, "gpu")) config.enable_gpu else if (std.mem.eql(u8, required_feature, "web")) config.enable_web else if (std.mem.eql(u8, required_feature, "monitoring")) config.enable_monitoring else false;

            if (!feature_enabled) {
                can_build = false;
                break;
            }
        }

        if (!can_build) continue;

        const exe = b.addExecutable(.{
            .name = example.name,
            .root_source_file = b.path(example.path),
            .target = target,
            .optimize = optimize,
        });
        exe.root_module.addImport("abi", abi_module);
        
        // Apply optimizations to examples
        if (optimize != .Debug) {
            exe.strip = config.strip_debug;
            exe.link_gc_sections = true;
        }

        const install_exe = b.addInstallArtifact(exe, .{
            .dest_dir = .{ .override = .{ .custom = "examples" } },
        });

        const example_step = b.step(
            b.fmt("example-{s}", .{example.name}),
            b.fmt("Build {s} example", .{example.name}),
        );
        example_step.dependOn(&install_exe.step);

        const run_example = b.addRunArtifact(exe);
        const run_step = b.step(
            b.fmt("run-{s}", .{example.name}),
            b.fmt("Run {s} example", .{example.name}),
        );
        run_step.dependOn(&run_example.step);
    }

    // Build all examples step
    const all_examples_step = b.step("examples", "Build all examples");
    for (examples) |example| {
        const step_name = b.fmt("example-{s}", .{example.name});
        if (b.top_level_steps.get(step_name)) |step| {
            all_examples_step.dependOn(&step.step);
        }
    }
}

fn buildBenchmarks(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    config: BuildConfig,
) void {
    const bench_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    bench_exe.root_module.addImport("abi", abi_module);
    
    // Benchmarks should always be optimized
    if (config.enable_lto) {
        bench_exe.want_lto = true;
    }

    const install_bench = b.addInstallArtifact(bench_exe, .{
        .dest_dir = .{ .override = .{ .custom = "bench" } },
    });

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
) void {
    // Documentation generator
    const docs_exe = b.addExecutable(.{
        .name = "docs_generator",
        .root_source_file = b.path("src/tools/docs_generator.zig"),
        .target = target,
        .optimize = optimize,
    });
    docs_exe.root_module.addImport("abi", abi_module);

    const run_docs = b.addRunArtifact(docs_exe);
    const docs_step = b.step("docs", "Generate API documentation");
    docs_step.dependOn(&run_docs.step);

    // Zig's built-in documentation
    const lib_step = b.addSharedLibrary(.{
        .name = "abi",
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    const install_docs = b.addInstallDirectory(.{
        .source_dir = lib_step.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs/api",
    });

    const docs_auto_step = b.step("docs-auto", "Generate Zig autodocs");
    docs_auto_step.dependOn(&install_docs.step);
    
    // Combined docs step
    const all_docs_step = b.step("docs-all", "Generate all documentation");
    all_docs_step.dependOn(&run_docs.step);
    all_docs_step.dependOn(&install_docs.step);
}

fn buildTools(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_module: *std.Build.Module,
    config: BuildConfig,
) void {
    const tools = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "performance_profiler", .path = "tools/performance_profiler.zig" },
        .{ .name = "static_analyzer", .path = "src/tools/static_analysis.zig" },
        .{ .name = "memory_tracker", .path = "src/tools/memory_tracker.zig" },
    };

    for (tools) |tool| {
        // Check if tool source exists
        std.fs.cwd().access(tool.path, .{}) catch continue;
        
        const tool_exe = b.addExecutable(.{
            .name = tool.name,
            .root_source_file = b.path(tool.path),
            .target = target,
            .optimize = optimize,
        });
        tool_exe.root_module.addImport("abi", abi_module);
        
        if (config.enable_lto and optimize == .ReleaseFast) {
            tool_exe.want_lto = true;
        }

        const install_tool = b.addInstallArtifact(tool_exe, .{
            .dest_dir = .{ .override = .{ .custom = "tools" } },
        });

        const tool_step = b.step(
            b.fmt("tool-{s}", .{tool.name}),
            b.fmt("Build {s} tool", .{tool.name}),
        );
        tool_step.dependOn(&install_tool.step);
    }

    const tools_step = b.step("tools", "Build all development tools");
    for (tools) |tool| {
        const step_name = b.fmt("tool-{s}", .{tool.name});
        if (b.top_level_steps.get(step_name)) |step| {
            tools_step.dependOn(&step.step);
        }
    }
}

fn addCustomSteps(b: *std.Build, config: BuildConfig) void {
    // Format check step
    const fmt_step = b.step("fmt", "Format source code");
    const fmt_cmd = b.addSystemCommand(&.{ "zig", "fmt", "--check", "src/" });
    fmt_step.dependOn(&fmt_cmd.step);

    // Lint step
    const lint_step = b.step("lint", "Run linting checks");
    lint_step.dependOn(fmt_step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addRemoveDirTree("zig-out");
    clean_step.dependOn(&clean_cmd.step);

    // Install step with all components
    const install_all_step = b.step("install-all", "Install all components");
    install_all_step.dependOn(b.getInstallStep());
    
    if (config.enable_examples) {
        if (b.top_level_steps.get("examples")) |step| {
            install_all_step.dependOn(&step.step);
        }
    }
    
    if (b.top_level_steps.get("tools")) |step| {
        install_all_step.dependOn(&step.step);
    }

    // CI step
    const ci_step = b.step("ci", "Run CI pipeline");
    ci_step.dependOn(fmt_step);
    
    if (config.enable_testing) {
        if (b.top_level_steps.get("test-all")) |step| {
            ci_step.dependOn(&step.step);
        }
    }
    
    if (config.enable_benchmarks) {
        if (b.top_level_steps.get("bench")) |step| {
            ci_step.dependOn(&step.step);
        }
    }
}

fn getBuildTimestamp(b: *std.Build) []const u8 {
    _ = b;
    // In a real implementation, this would generate the actual timestamp
    return "2024-10-16T12:00:00Z";
}

fn getGitCommit(b: *std.Build) []const u8 {
    _ = b;
    // In a real implementation, this would run `git rev-parse HEAD`
    return "unknown";
}