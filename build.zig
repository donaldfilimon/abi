const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileError(std.fmt.comptimePrint(
            "ABI requires Zig 0.16.0 or newer (detected {d}.{d}.{d}).\nUse ./zigw <command> in this repository.",
            .{
                builtin.zig_version.major,
                builtin.zig_version.minor,
                builtin.zig_version.patch,
            },
        ));
    }
}

// ============================================================================
// GPU Backend Configuration
// ============================================================================

pub const GpuBackend = enum {
    none,
    auto,
    cuda,
    vulkan,
    stdgpu,
    metal,
    webgpu,
    opengl,
    opengles,
    webgl2,
    fpga,

    pub fn fromString(s: []const u8) ?GpuBackend {
        return std.StaticStringMap(GpuBackend).initComptime(.{
            .{ "none", .none },     .{ "auto", .auto },     .{ "cuda", .cuda },
            .{ "vulkan", .vulkan }, .{ "stdgpu", .stdgpu }, .{ "metal", .metal },
            .{ "webgpu", .webgpu }, .{ "opengl", .opengl }, .{ "opengles", .opengles },
            .{ "webgl2", .webgl2 }, .{ "fpga", .fpga },
        }).get(s);
    }
};

fn parseGpuBackends(b: *std.Build, enable_gpu: bool, enable_web: bool) []const GpuBackend {
    const backend_str = b.option([]const u8, "gpu-backend", "GPU backend(s): auto, none, cuda, vulkan, metal, webgpu, opengl, opengles, webgl2, stdgpu, fpga (comma-separated)");

    // Legacy flags (deprecated)
    const legacy = .{
        .cuda = b.option(bool, "gpu-cuda", "Enable CUDA (deprecated: use -Dgpu-backend=cuda)"),
        .vulkan = b.option(bool, "gpu-vulkan", "Enable Vulkan (deprecated: use -Dgpu-backend=vulkan)"),
        .stdgpu = b.option(bool, "gpu-stdgpu", "Enable std.gpu (deprecated: use -Dgpu-backend=stdgpu)"),
        .metal = b.option(bool, "gpu-metal", "Enable Metal (deprecated: use -Dgpu-backend=metal)"),
        .webgpu = b.option(bool, "gpu-webgpu", "Enable WebGPU (deprecated: use -Dgpu-backend=webgpu)"),
        .opengl = b.option(bool, "gpu-opengl", "Enable OpenGL (deprecated: use -Dgpu-backend=opengl)"),
        .opengles = b.option(bool, "gpu-opengles", "Enable OpenGL ES (deprecated: use -Dgpu-backend=opengles)"),
        .webgl2 = b.option(bool, "gpu-webgl2", "Enable WebGL2 (deprecated: use -Dgpu-backend=webgl2)"),
        .fpga = b.option(bool, "gpu-fpga", "Enable FPGA (deprecated: use -Dgpu-backend=fpga)"),
    };

    const has_legacy = legacy.cuda != null or legacy.vulkan != null or legacy.stdgpu != null or
        legacy.metal != null or legacy.webgpu != null or legacy.opengl != null or
        legacy.opengles != null or legacy.webgl2 != null or legacy.fpga != null;

    if (has_legacy) std.log.warn("Legacy GPU flags are deprecated. Use -Dgpu-backend=cuda,vulkan instead.", .{});

    const backend_count = @typeInfo(GpuBackend).@"enum".fields.len;
    var buffer: [backend_count]GpuBackend = undefined;
    var seen = [_]bool{false} ** backend_count;
    var count: usize = 0;
    var use_auto = false;

    const addBackend = struct {
        fn call(backend: GpuBackend, buffer_ptr: *[backend_count]GpuBackend, count_ptr: *usize, seen_ptr: *[backend_count]bool) void {
            const idx = @intFromEnum(backend);
            if (seen_ptr[idx]) return;
            if (count_ptr.* >= buffer_ptr.len) return;
            buffer_ptr[count_ptr.*] = backend;
            count_ptr.* += 1;
            seen_ptr[idx] = true;
        }
    }.call;

    if (backend_str) |str| {
        var iter = std.mem.splitScalar(u8, str, ',');
        while (iter.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t");
            if (trimmed.len == 0) continue;
            if (GpuBackend.fromString(trimmed)) |backend| {
                if (backend == .none) return &.{};
                if (backend == .auto) {
                    use_auto = true;
                    continue;
                }
                addBackend(backend, &buffer, &count, &seen);
            } else std.log.warn("Unknown GPU backend: '{s}'", .{trimmed});
        }
        if (use_auto) {
            if (enable_gpu) addBackend(.vulkan, &buffer, &count, &seen);
            if (enable_web) {
                addBackend(.webgpu, &buffer, &count, &seen);
                addBackend(.webgl2, &buffer, &count, &seen);
            }
        }
    } else {
        // Legacy defaults
        if (legacy.cuda orelse false) addBackend(.cuda, &buffer, &count, &seen);
        if (legacy.vulkan orelse enable_gpu) addBackend(.vulkan, &buffer, &count, &seen);
        if (legacy.stdgpu orelse false) addBackend(.stdgpu, &buffer, &count, &seen);
        if (legacy.metal orelse false) addBackend(.metal, &buffer, &count, &seen);
        if (legacy.webgpu orelse enable_web) addBackend(.webgpu, &buffer, &count, &seen);
        if (legacy.opengl orelse false) addBackend(.opengl, &buffer, &count, &seen);
        if (legacy.opengles orelse false) addBackend(.opengles, &buffer, &count, &seen);
        if (legacy.webgl2 orelse enable_web) addBackend(.webgl2, &buffer, &count, &seen);
        if (legacy.fpga orelse false) addBackend(.fpga, &buffer, &count, &seen);
    }
    return b.allocator.dupe(GpuBackend, buffer[0..count]) catch &.{};
}

// ============================================================================
// Build Options
// ============================================================================

const BuildOptions = struct {
    enable_gpu: bool,
    enable_ai: bool,
    enable_explore: bool,
    enable_llm: bool,
    enable_vision: bool,
    enable_web: bool,
    enable_database: bool,
    enable_network: bool,
    enable_profiling: bool,
    enable_analytics: bool,
    gpu_backends: []const GpuBackend,

    pub fn hasGpuBackend(self: BuildOptions, backend: GpuBackend) bool {
        for (self.gpu_backends) |b| if (b == backend) return true;
        return false;
    }

    pub fn hasAnyGpuBackend(self: BuildOptions, backends: []const GpuBackend) bool {
        for (backends) |check| if (self.hasGpuBackend(check)) return true;
        return false;
    }

    // Legacy accessors for backward compatibility
    pub fn gpu_cuda(self: BuildOptions) bool {
        return self.hasGpuBackend(.cuda);
    }
    pub fn gpu_vulkan(self: BuildOptions) bool {
        return self.hasGpuBackend(.vulkan);
    }
    pub fn gpu_stdgpu(self: BuildOptions) bool {
        return self.hasGpuBackend(.stdgpu);
    }
    pub fn gpu_metal(self: BuildOptions) bool {
        return self.hasGpuBackend(.metal);
    }
    pub fn gpu_webgpu(self: BuildOptions) bool {
        return self.hasGpuBackend(.webgpu);
    }
    pub fn gpu_opengl(self: BuildOptions) bool {
        return self.hasGpuBackend(.opengl);
    }
    pub fn gpu_opengles(self: BuildOptions) bool {
        return self.hasGpuBackend(.opengles);
    }
    pub fn gpu_webgl2(self: BuildOptions) bool {
        return self.hasGpuBackend(.webgl2);
    }
    pub fn gpu_fpga(self: BuildOptions) bool {
        return self.hasGpuBackend(.fpga);
    }
};

fn readBuildOptions(b: *std.Build) BuildOptions {
    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU support") orelse true;
    const enable_ai = b.option(bool, "enable-ai", "Enable AI features") orelse true;
    const enable_web = b.option(bool, "enable-web", "Enable web features") orelse true;

    return .{
        .enable_gpu = enable_gpu,
        .enable_ai = enable_ai,
        .enable_explore = b.option(bool, "enable-explore", "Enable AI code exploration") orelse enable_ai,
        .enable_llm = b.option(bool, "enable-llm", "Enable local LLM inference") orelse enable_ai,
        .enable_vision = b.option(bool, "enable-vision", "Enable vision/image processing") orelse enable_ai,
        .enable_web = enable_web,
        .enable_database = b.option(bool, "enable-database", "Enable database features") orelse true,
        .enable_network = b.option(bool, "enable-network", "Enable network distributed compute") orelse true,
        .enable_profiling = b.option(bool, "enable-profiling", "Enable profiling and metrics") orelse true,
        .enable_analytics = b.option(bool, "enable-analytics", "Enable analytics event tracking") orelse true,
        .gpu_backends = parseGpuBackends(b, enable_gpu, enable_web),
    };
}

fn validateOptions(options: BuildOptions) void {
    const has_native = options.hasAnyGpuBackend(&.{ .cuda, .vulkan, .stdgpu, .metal, .opengl, .opengles });
    const has_web = options.hasAnyGpuBackend(&.{ .webgpu, .webgl2 });

    if (has_native and !options.enable_gpu)
        std.log.err("GPU backends enabled but enable-gpu=false", .{});
    if (has_web and !options.enable_web)
        std.log.err("Web GPU backends enabled but enable-web=false", .{});
    if (options.hasGpuBackend(.cuda) and options.hasGpuBackend(.vulkan))
        std.log.warn("Both CUDA and Vulkan backends enabled; may cause conflicts", .{});
    if (options.hasGpuBackend(.opengl) and options.hasGpuBackend(.webgl2))
        std.log.warn("Both OpenGL and WebGL2 enabled; prefer one", .{});
    if (options.hasGpuBackend(.opengl) and options.hasGpuBackend(.opengles))
        std.log.warn("Both OpenGL and OpenGL ES enabled; typically mutually exclusive", .{});
}

// ============================================================================
// Feature Flag Validation Matrix
// ============================================================================

/// Compact flag combination for validation. Sub-feature flags (explore, llm,
/// vision) inherit from enable_ai.
const FlagCombo = struct {
    name: []const u8,
    enable_ai: bool = false,
    enable_gpu: bool = false,
    enable_web: bool = false,
    enable_database: bool = false,
    enable_network: bool = false,
    enable_profiling: bool = false,
    enable_analytics: bool = false,
};

/// Critical flag combinations that must compile. Covers: all on, all off,
/// each feature solo, and each feature disabled with the rest enabled.
const validation_matrix = [_]FlagCombo{
    .{ .name = "all-enabled", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true },
    .{ .name = "all-disabled" },
    .{ .name = "ai-only", .enable_ai = true },
    .{ .name = "gpu-only", .enable_gpu = true },
    .{ .name = "web-only", .enable_web = true },
    .{ .name = "database-only", .enable_database = true },
    .{ .name = "network-only", .enable_network = true },
    .{ .name = "profiling-only", .enable_profiling = true },
    .{ .name = "analytics-only", .enable_analytics = true },
    .{ .name = "no-ai", .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true },
    .{ .name = "no-gpu", .enable_ai = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true },
    .{ .name = "no-web", .enable_ai = true, .enable_gpu = true, .enable_database = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true },
    .{ .name = "no-database", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_network = true, .enable_profiling = true, .enable_analytics = true },
    .{ .name = "no-network", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_profiling = true, .enable_analytics = true },
    .{ .name = "no-profiling", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_analytics = true },
    .{ .name = "no-analytics", .enable_ai = true, .enable_gpu = true, .enable_web = true, .enable_database = true, .enable_network = true, .enable_profiling = true },
};

fn comboToBuildOptions(combo: FlagCombo) BuildOptions {
    return .{
        .enable_ai = combo.enable_ai,
        .enable_gpu = combo.enable_gpu,
        .enable_explore = combo.enable_ai,
        .enable_llm = combo.enable_ai,
        .enable_vision = combo.enable_ai,
        .enable_web = combo.enable_web,
        .enable_database = combo.enable_database,
        .enable_network = combo.enable_network,
        .enable_profiling = combo.enable_profiling,
        .enable_analytics = combo.enable_analytics,
        .gpu_backends = if (combo.enable_gpu) &.{.vulkan} else &.{},
    };
}

fn addFlagValidation(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const validate_step = b.step(
        "validate-flags",
        "Compile with every critical feature-flag combination",
    );

    inline for (validation_matrix) |combo| {
        const opts = comboToBuildOptions(combo);
        const build_opts_mod = createBuildOptionsModule(b, opts);
        const abi_mod = b.createModule(.{
            .root_source_file = b.path("src/abi.zig"),
            .target = target,
            .optimize = optimize,
        });
        abi_mod.addImport("build_options", build_opts_mod);

        // Static library compile-check: verifies code compiles without linking
        const check = b.addLibrary(.{
            .name = "validate-" ++ combo.name,
            .root_module = abi_mod,
            .linkage = .static,
        });
        validate_step.dependOn(&check.step);
    }

    return validate_step;
}

// ============================================================================
// Table-Driven Build Targets
// ============================================================================

const BuildTarget = struct {
    name: []const u8,
    step_name: []const u8,
    description: []const u8,
    source_path: []const u8,
    optimize: ?std.builtin.OptimizeMode = null,
};

const example_targets = [_]BuildTarget{
    .{ .name = "example-hello", .step_name = "run-hello", .description = "Run hello example", .source_path = "examples/hello.zig" },
    .{ .name = "example-database", .step_name = "run-database", .description = "Run database example", .source_path = "examples/database.zig" },
    .{ .name = "example-agent", .step_name = "run-agent", .description = "Run agent example", .source_path = "examples/agent.zig" },
    .{ .name = "example-compute", .step_name = "run-compute", .description = "Run compute example", .source_path = "examples/compute.zig" },
    .{ .name = "example-network", .step_name = "run-network", .description = "Run network example", .source_path = "examples/network.zig" },
    .{ .name = "example-discord", .step_name = "run-discord", .description = "Run discord example", .source_path = "examples/discord.zig" },
    .{ .name = "example-llm", .step_name = "run-llm", .description = "Run LLM example", .source_path = "examples/llm.zig" },
    .{ .name = "example-training", .step_name = "run-training", .description = "Run training example", .source_path = "examples/training.zig" },
    .{ .name = "example-ha", .step_name = "run-ha", .description = "Run HA example", .source_path = "examples/ha.zig" },
    .{ .name = "example-train-demo", .step_name = "run-train-demo", .description = "Run LLM training demo", .source_path = "examples/training/train_demo.zig" },
    .{ .name = "example-orchestration", .step_name = "run-orchestration", .description = "Run multi-model orchestration example", .source_path = "examples/orchestration.zig" },
    .{ .name = "example-train-ava", .step_name = "run-train-ava", .description = "Train Ava assistant from gpt-oss", .source_path = "examples/train_ava.zig" },
    .{ .name = "example-concurrency", .step_name = "run-concurrency", .description = "Run concurrency primitives example", .source_path = "examples/concurrency.zig" },
    .{ .name = "example-observability", .step_name = "run-observability", .description = "Run observability example", .source_path = "examples/observability.zig" },
    .{ .name = "example-gpu", .step_name = "run-gpu", .description = "Run GPU example", .source_path = "examples/gpu.zig" },
    .{ .name = "example-streaming", .step_name = "run-streaming", .description = "Run streaming API example", .source_path = "examples/streaming.zig" },
    .{ .name = "example-registry", .step_name = "run-registry", .description = "Run feature registry example", .source_path = "examples/registry.zig" },
    .{ .name = "example-embeddings", .step_name = "run-embeddings", .description = "Run embeddings example", .source_path = "examples/embeddings.zig" },
    .{ .name = "example-config", .step_name = "run-config", .description = "Run configuration example", .source_path = "examples/config.zig" },
    .{ .name = "example-tensor-ops", .step_name = "run-tensor-ops", .description = "Run tensor + matrix + SIMD example", .source_path = "examples/tensor_ops.zig" },
    .{ .name = "example-concurrent-pipeline", .step_name = "run-concurrent-pipeline", .description = "Run channel + thread pool + DAG pipeline example", .source_path = "examples/concurrent_pipeline.zig" },
};

const benchmark_targets = [_]BuildTarget{
    .{ .name = "benchmarks", .step_name = "benchmarks", .description = "Run comprehensive benchmarks", .source_path = "benchmarks/main.zig", .optimize = .ReleaseFast },
    .{ .name = "bench-competitive", .step_name = "bench-competitive", .description = "Run competitive benchmarks", .source_path = "benchmarks/run_competitive.zig", .optimize = .ReleaseFast },
};

fn pathExists(b: *std.Build, path: []const u8) bool {
    if (builtin.zig_version.minor >= 16) {
        // Zig 0.16+: access(io, sub_path, flags)
        b.build_root.handle.access(b.graph.io, path, .{}) catch return false;
    } else {
        // Zig 0.15: access(sub_path, flags)
        b.build_root.handle.access(path, .{}) catch return false;
    }
    return true;
}

fn buildTargets(
    b: *std.Build,
    targets: []const BuildTarget,
    abi_module: *std.Build.Module,
    build_opts: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    aggregate: ?*std.Build.Step,
    aggregate_runs: bool,
) void {
    for (targets) |t| {
        if (!pathExists(b, t.source_path)) continue;
        const exe_optimize = t.optimize orelse optimize;
        const exe = b.addExecutable(.{
            .name = t.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(t.source_path),
                .target = target,
                .optimize = exe_optimize,
                .link_libc = true,
            }),
        });
        exe.root_module.addImport("abi", abi_module);
        exe.root_module.addImport("build_options", build_opts);

        // Apply performance optimizations
        applyPerformanceTweaks(exe, exe_optimize);

        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);
        const step = b.step(t.step_name, t.description);
        step.dependOn(&run.step);
        if (aggregate) |agg| {
            if (aggregate_runs) {
                agg.dependOn(&run.step);
            } else {
                agg.dependOn(&exe.step);
            }
        }
    }
}

// ============================================================================
// Module Creation
// ============================================================================

fn createBuildOptionsModule(b: *std.Build, options: BuildOptions) *std.Build.Module {
    var opts = b.addOptions();
    opts.addOption([]const u8, "package_version", "0.4.0");
    opts.addOption(bool, "enable_gpu", options.enable_gpu);
    opts.addOption(bool, "enable_ai", options.enable_ai);
    opts.addOption(bool, "enable_explore", options.enable_explore);
    opts.addOption(bool, "enable_llm", options.enable_llm);
    opts.addOption(bool, "enable_vision", options.enable_vision);
    opts.addOption(bool, "enable_web", options.enable_web);
    opts.addOption(bool, "enable_database", options.enable_database);
    opts.addOption(bool, "enable_network", options.enable_network);
    opts.addOption(bool, "enable_profiling", options.enable_profiling);
    opts.addOption(bool, "enable_analytics", options.enable_analytics);
    opts.addOption(bool, "gpu_cuda", options.gpu_cuda());
    opts.addOption(bool, "gpu_vulkan", options.gpu_vulkan());
    opts.addOption(bool, "gpu_stdgpu", options.gpu_stdgpu());
    opts.addOption(bool, "gpu_metal", options.gpu_metal());
    opts.addOption(bool, "gpu_webgpu", options.gpu_webgpu());
    opts.addOption(bool, "gpu_opengl", options.gpu_opengl());
    opts.addOption(bool, "gpu_opengles", options.gpu_opengles());
    opts.addOption(bool, "gpu_webgl2", options.gpu_webgl2());
    opts.addOption(bool, "gpu_fpga", options.gpu_fpga());
    return opts.createModule();
}

fn createCliModule(b: *std.Build, abi_module: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const cli = b.createModule(.{ .root_source_file = b.path("tools/cli/mod.zig"), .target = target, .optimize = optimize });
    cli.addImport("abi", abi_module);
    return cli;
}

fn createAbiModule(b: *std.Build, options: BuildOptions, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const build_opts = createBuildOptionsModule(b, options);
    // Use createModule (anonymous) to avoid double-registering "abi" in the global namespace.
    // The primary "abi" module is registered in build() via addModule.
    const abi = b.createModule(.{ .root_source_file = b.path("src/abi.zig"), .target = target, .optimize = optimize });
    abi.addImport("build_options", build_opts);
    return abi;
}

// ============================================================================
// Performance Tuning
// ============================================================================

fn applyPerformanceTweaks(exe: *std.Build.Step.Compile, optimize: std.builtin.OptimizeMode) void {
    if (optimize == .ReleaseFast or optimize == .ReleaseSmall) {
        // Link Time Optimization: significantly improves throughput and reduces binary size
        // by allowing optimizations across module boundaries.
        // exe.want_lto = true; // Removed as it is not a valid field in Zig 0.16 Build.Step.Compile

        // Stripping: Reduces binary size, improving disk resource utilization and start-up latency.
        // We default to true for release builds unless explicitly overridden later (e.g. for profiling).
        if (exe.root_module.strip == null) {
            exe.root_module.strip = true;
        }
    }
}

// ============================================================================
// Main Build Function
// ============================================================================

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const options = readBuildOptions(b);
    validateOptions(options);

    const build_opts = createBuildOptionsModule(b, options);
    const abi_module = b.addModule("abi", .{ .root_source_file = b.path("src/abi.zig"), .target = target, .optimize = optimize });
    abi_module.addImport("build_options", build_opts);

    // CLI executable
    const cli_path = if (pathExists(b, "tools/cli/main.zig")) "tools/cli/main.zig" else "src/api/main.zig";
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{ .root_source_file = b.path(cli_path), .target = target, .optimize = optimize, .link_libc = true }),
    });
    exe.root_module.addImport("abi", abi_module);
    if (pathExists(b, "tools/cli/main.zig")) exe.root_module.addImport("cli", createCliModule(b, abi_module, target, optimize));

    // Apply performance optimizations (LTO, strip)
    applyPerformanceTweaks(exe, optimize);

    b.installArtifact(exe);

    const run_cli = b.addRunArtifact(exe);
    if (b.args) |args| run_cli.addArgs(args);
    b.step("run", "Run the ABI CLI").dependOn(&run_cli.step);

    // Examples and benchmarks (table-driven)
    buildTargets(b, &example_targets, abi_module, build_opts, target, optimize, b.step("examples", "Build all examples"), false);

    // ---------------------------------------------------------------------------
    // CLI smoke-test step (runs all example commands sequentially)
    // Cross-platform: directly invokes the built CLI binary
    // ---------------------------------------------------------------------------
    const cli_tests_step = b.step("cli-tests", "Run smoke test of CLI commands");
    const cli_commands = [_][]const []const u8{
        &.{"--help"},
        &.{"--version"},
        &.{"system-info"},
        &.{ "db", "stats" },
        &.{ "gpu", "status" },
        &.{ "task", "list" },
        &.{ "config", "show" },
    };
    for (cli_commands) |args| {
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.addArgs(args);
        // Smoke test: just verify commands run without crashing (any exit code is OK)
        cli_tests_step.dependOn(&run_cmd.step);
    }

    // ---------------------------------------------------------------------------
    // Lint step (formatting check)
    // ---------------------------------------------------------------------------
    const lint_fmt = b.addFmt(.{
        .paths = &.{"."},
        .check = true,
    });
    b.step("lint", "Check code formatting").dependOn(&lint_fmt.step);

    // ---------------------------------------------------------------------------
    // Tests - defined before full-check to allow step dependency
    // ---------------------------------------------------------------------------
    var test_step: ?*std.Build.Step = null;
    if (pathExists(b, "src/services/tests/mod.zig")) {
        const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("src/services/tests/mod.zig"), .target = target, .optimize = optimize, .link_libc = true }) });
        tests.root_module.addImport("abi", abi_module);
        tests.root_module.addImport("build_options", build_opts);
        b.step("typecheck", "Compile tests without running").dependOn(&tests.step);
        const run_tests = b.addRunArtifact(tests);
        run_tests.skip_foreign_checks = true;
        test_step = b.step("test", "Run unit tests");
        test_step.?.dependOn(&run_tests.step);
    }

    // ---------------------------------------------------------------------------
    // Feature flag validation matrix
    // ---------------------------------------------------------------------------
    const validate_flags_step = addFlagValidation(b, target, optimize);

    // ---------------------------------------------------------------------------
    // Full verification step – formatting, tests, CLI smoke test, flag validation
    // Cross-platform: chains format check, tests, CLI smoke tests, and flag matrix
    // ---------------------------------------------------------------------------
    const full_check_step = b.step("full-check", "Run formatting, unit tests, CLI smoke tests, and flag validation");
    full_check_step.dependOn(&lint_fmt.step);
    if (test_step) |ts| {
        full_check_step.dependOn(ts);
    }
    full_check_step.dependOn(cli_tests_step);
    full_check_step.dependOn(validate_flags_step);
    buildTargets(b, &benchmark_targets, abi_module, build_opts, target, optimize, b.step("bench-all", "Run all benchmark suites"), true);

    // Documentation - API markdown generation
    if (pathExists(b, "tools/gendocs/main.zig")) {
        const gendocs = b.addExecutable(.{ .name = "gendocs", .root_module = b.createModule(.{ .root_source_file = b.path("tools/gendocs/main.zig"), .target = target, .optimize = optimize, .link_libc = true }) });
        const run_gendocs = b.addRunArtifact(gendocs);
        if (b.args) |args| run_gendocs.addArgs(args);
        b.step("gendocs", "Generate API documentation").dependOn(&run_gendocs.step);
    }

    // Documentation - Static site generation
    if (pathExists(b, "tools/docs_site/main.zig")) {
        const docs_site = b.addExecutable(.{
            .name = "docs-site",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/docs_site/main.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        const run_docs_site = b.addRunArtifact(docs_site);
        if (b.args) |args| run_docs_site.addArgs(args);
        b.step("docs-site", "Generate documentation website").dependOn(&run_docs_site.step);
    }

    // Profile build
    if (pathExists(b, "tools/cli/main.zig")) {
        var profile_opts = options;
        profile_opts.enable_profiling = true;
        const abi_profile = createAbiModule(b, profile_opts, target, optimize);
        const profile_exe = b.addExecutable(.{ .name = "abi-profile", .root_module = b.createModule(.{ .root_source_file = b.path("tools/cli/main.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true }) });
        profile_exe.root_module.addImport("abi", abi_profile);
        profile_exe.root_module.addImport("cli", createCliModule(b, abi_profile, target, optimize));

        // Profiling specific overrides:
        // 1. Keep symbols for profilers (don't strip)
        profile_exe.root_module.strip = false;
        // 2. Keep frame pointers for accurate stack unwinding in perf/instruments
        profile_exe.root_module.omit_frame_pointer = false;

        b.installArtifact(profile_exe);
        b.step("profile", "Build with performance profiling").dependOn(b.getInstallStep());
    }

    // Mobile
    const mobile_step = b.step("mobile", "Build for mobile targets (Android/iOS)");
    const enable_mobile = b.option(bool, "enable-mobile", "Enable mobile target cross-compilation") orelse false;

    if (enable_mobile) {
        // Android (aarch64)
        const android_target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .android });
        const abi_android = b.addLibrary(.{
            .name = "abi-android",
            .root_module = b.createModule(.{ .root_source_file = b.path("src/abi.zig"), .target = android_target, .optimize = optimize }),
            .linkage = .static,
        });
        abi_android.root_module.addImport("build_options", createBuildOptionsModule(b, options));
        mobile_step.dependOn(&b.addInstallArtifact(abi_android, .{ .dest_dir = .{ .override = .{ .custom = "mobile/android" } } }).step);

        // iOS (aarch64) - Simulated as macOS-none for now or actual ios if SDK present
        // Zig treats aarch64-macos as compatible for general logic, but strict iOS requires ios tag
        const ios_target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .ios });
        const abi_ios = b.addLibrary(.{
            .name = "abi-ios",
            .root_module = b.createModule(.{ .root_source_file = b.path("src/abi.zig"), .target = ios_target, .optimize = optimize }),
            .linkage = .static,
        });
        abi_ios.root_module.addImport("build_options", createBuildOptionsModule(b, options));
        mobile_step.dependOn(&b.addInstallArtifact(abi_ios, .{ .dest_dir = .{ .override = .{ .custom = "mobile/ios" } } }).step);
    }

    // C Library (Shared Object / DLL)
    if (pathExists(b, "bindings/c/src/abi_c.zig")) {
        const lib = b.addLibrary(.{
            .name = "abi",
            .root_module = b.createModule(.{
                .root_source_file = b.path("bindings/c/src/abi_c.zig"),
                .target = target,
                .optimize = optimize,
            }),
            .linkage = .dynamic,
        });
        lib.root_module.addImport("abi", abi_module);
        lib.root_module.addImport("build_options", build_opts);

        const lib_install = b.addInstallArtifact(lib, .{});
        b.step("lib", "Build C shared library").dependOn(&lib_install.step);

        // Install C header file
        const header_install = b.addInstallFile(b.path("bindings/c/include/abi.h"), "include/abi.h");
        b.step("c-header", "Install C header file").dependOn(&header_install.step);
    }

    // Performance Verification Tool (build only - requires piped input to run)
    // Usage: zig build bench-competitive -- --json | ./zig-out/bin/abi-check-perf
    if (pathExists(b, "tools/perf/check.zig")) {
        const check_perf_exe = b.addExecutable(.{
            .name = "abi-check-perf",
            .root_module = b.createModule(.{ .root_source_file = b.path("tools/perf/check.zig"), .target = target, .optimize = .ReleaseSafe }),
        });
        const install_check_perf = b.addInstallArtifact(check_perf_exe, .{});
        b.step("check-perf", "Build performance verification tool (pipe benchmark JSON to run)").dependOn(&install_check_perf.step);
    }

    // WASM - only build if bindings exist (removed for reimplementation)
    if (pathExists(b, "bindings/wasm/abi_wasm.zig")) {
        const wasm_target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
        var wasm_opts = options;
        wasm_opts.enable_database = false;
        wasm_opts.enable_network = false;
        wasm_opts.enable_gpu = false;
        wasm_opts.enable_profiling = false;
        wasm_opts.enable_web = false;
        wasm_opts.gpu_backends = &.{};

        const wasm_build_opts = createBuildOptionsModule(b, wasm_opts);
        const abi_wasm = b.addModule("abi-wasm", .{ .root_source_file = b.path("src/abi.zig"), .target = wasm_target, .optimize = optimize });
        abi_wasm.addImport("build_options", wasm_build_opts);

        const wasm_lib = b.addExecutable(.{ .name = "abi", .root_module = b.createModule(.{ .root_source_file = b.path("bindings/wasm/abi_wasm.zig"), .target = wasm_target, .optimize = optimize }) });
        wasm_lib.entry = .disabled;
        wasm_lib.rdynamic = true;
        wasm_lib.root_module.addImport("abi", abi_wasm);

        b.step("check-wasm", "Check WASM compilation").dependOn(&wasm_lib.step);
        b.step("wasm", "Build WASM bindings").dependOn(&b.addInstallArtifact(wasm_lib, .{ .dest_dir = .{ .override = .{ .custom = "wasm" } } }).step);
    } else {
        // WASM bindings not available - steps are no-ops
        _ = b.step("check-wasm", "Check WASM compilation (bindings not available)");
        _ = b.step("wasm", "Build WASM bindings (bindings not available)");
    }
}
