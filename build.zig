const std = @import("std");

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

    var buffer: [11]GpuBackend = undefined;
    var count: usize = 0;

    if (backend_str) |str| {
        var iter = std.mem.splitScalar(u8, str, ',');
        while (iter.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " \t");
            if (trimmed.len == 0) continue;
            if (GpuBackend.fromString(trimmed)) |backend| {
                if (backend == .none) return &.{};
                if (count < buffer.len) {
                    buffer[count] = backend;
                    count += 1;
                }
            } else std.log.warn("Unknown GPU backend: '{s}'", .{trimmed});
        }
    } else {
        // Legacy defaults
        if (legacy.cuda orelse false) {
            buffer[count] = .cuda;
            count += 1;
        }
        if (legacy.vulkan orelse enable_gpu) {
            buffer[count] = .vulkan;
            count += 1;
        }
        if (legacy.stdgpu orelse false) {
            buffer[count] = .stdgpu;
            count += 1;
        }
        if (legacy.metal orelse false) {
            buffer[count] = .metal;
            count += 1;
        }
        if (legacy.webgpu orelse enable_web) {
            buffer[count] = .webgpu;
            count += 1;
        }
        if (legacy.opengl orelse false) {
            buffer[count] = .opengl;
            count += 1;
        }
        if (legacy.opengles orelse false) {
            buffer[count] = .opengles;
            count += 1;
        }
        if (legacy.webgl2 orelse enable_web) {
            buffer[count] = .webgl2;
            count += 1;
        }
        if (legacy.fpga orelse false) {
            buffer[count] = .fpga;
            count += 1;
        }
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
};

const benchmark_targets = [_]BuildTarget{
    .{ .name = "benchmarks", .step_name = "benchmarks", .description = "Run comprehensive benchmarks", .source_path = "benchmarks/run.zig", .optimize = .ReleaseFast },
    .{ .name = "bench-competitive", .step_name = "bench-competitive", .description = "Run competitive benchmarks", .source_path = "benchmarks/run_competitive.zig", .optimize = .ReleaseFast },
    .{ .name = "abi-benchmark", .step_name = "benchmark-legacy", .description = "Run legacy benchmarks", .source_path = "benchmarks/legacy.zig", .optimize = .ReleaseFast },
};

fn pathExists(path: []const u8) bool {
    const file = std.Io.Dir.cwd().openFile(std.Options.debug_io, path, .{}) catch return false;
    file.close(std.Options.debug_io);
    return true;
}

fn buildTargets(b: *std.Build, targets: []const BuildTarget, abi_module: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, aggregate: ?*std.Build.Step) void {
    for (targets) |t| {
        if (!pathExists(t.source_path)) continue;
        const exe = b.addExecutable(.{
            .name = t.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(t.source_path),
                .target = target,
                .optimize = t.optimize orelse optimize,
                .link_libc = true,
            }),
        });
        exe.root_module.addImport("abi", abi_module);
        b.installArtifact(exe);

        const run = b.addRunArtifact(exe);
        if (b.args) |args| run.addArgs(args);
        const step = b.step(t.step_name, t.description);
        step.dependOn(b.getInstallStep());
        step.dependOn(&run.step);
        if (aggregate) |agg| agg.dependOn(&exe.step);
    }
}

// ============================================================================
// Module Creation
// ============================================================================

fn createBuildOptionsModule(b: *std.Build, options: BuildOptions) *std.Build.Module {
    var opts = b.addOptions();
    opts.addOption([]const u8, "package_version", "0.1.1");
    opts.addOption(bool, "enable_gpu", options.enable_gpu);
    opts.addOption(bool, "enable_ai", options.enable_ai);
    opts.addOption(bool, "enable_explore", options.enable_explore);
    opts.addOption(bool, "enable_llm", options.enable_llm);
    opts.addOption(bool, "enable_vision", options.enable_vision);
    opts.addOption(bool, "enable_web", options.enable_web);
    opts.addOption(bool, "enable_database", options.enable_database);
    opts.addOption(bool, "enable_network", options.enable_network);
    opts.addOption(bool, "enable_profiling", options.enable_profiling);
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
    const abi = b.addModule("abi", .{ .root_source_file = b.path("src/abi.zig"), .target = target, .optimize = optimize });
    abi.addImport("build_options", build_opts);
    return abi;
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
    const cli_path = if (pathExists("tools/cli/main.zig")) "tools/cli/main.zig" else "src/main.zig";
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{ .root_source_file = b.path(cli_path), .target = target, .optimize = optimize, .link_libc = true }),
    });
    exe.root_module.addImport("abi", abi_module);
    if (pathExists("tools/cli/main.zig")) exe.root_module.addImport("cli", createCliModule(b, abi_module, target, optimize));
    b.installArtifact(exe);

    const run_cli = b.addRunArtifact(exe);
    run_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cli.addArgs(args);
    b.step("run", "Run the ABI CLI").dependOn(&run_cli.step);

    // Examples and benchmarks (table-driven)
    buildTargets(b, &example_targets, abi_module, target, optimize, b.step("examples", "Build all examples"));

    // ---------------------------------------------------------------------------
    // CLI smoke-test step (runs all example commands sequentially)
    // ---------------------------------------------------------------------------
    const cli_test_cmd = b.addSystemCommand(&[_][]const u8{ "cmd", "/c", "scripts\\run_cli_tests.bat" });
    b.step("cli-tests", "Run smoke test of all CLI example commands").dependOn(&cli_test_cmd.step);

    // ---------------------------------------------------------------------------
    // Full verification step – formatting, tests, CLI smoke test, benchmarks
    // ---------------------------------------------------------------------------
    const full_check_cmd = b.addSystemCommand(&[_][]const u8{ "cmd", "/c", "scripts\\full_check.bat" });
    b.step("full-check", "Run formatting, unit tests, CLI smoke tests, and benchmarks").dependOn(&full_check_cmd.step);
    buildTargets(b, &benchmark_targets, abi_module, target, optimize, b.step("bench-all", "Run all benchmark suites"));

    // Tests
    if (pathExists("src/tests/mod.zig")) {
        const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("src/tests/mod.zig"), .target = target, .optimize = optimize, .link_libc = true }) });
        tests.root_module.addImport("abi", abi_module);
        tests.root_module.addImport("build_options", build_opts);
        const run_tests = b.addRunArtifact(tests);
        run_tests.skip_foreign_checks = true;
        b.step("test", "Run unit tests").dependOn(&run_tests.step);
    }

    // Documentation - API markdown generation
    if (pathExists("tools/gendocs.zig")) {
        const gendocs = b.addExecutable(.{ .name = "gendocs", .root_module = b.createModule(.{ .root_source_file = b.path("tools/gendocs.zig"), .target = target, .optimize = optimize, .link_libc = true }) });
        const run_gendocs = b.addRunArtifact(gendocs);
        if (b.args) |args| run_gendocs.addArgs(args);
        b.step("gendocs", "Generate API documentation").dependOn(&run_gendocs.step);
    }

    // Documentation - Static site generation
    if (pathExists("tools/docgen/main.zig")) {
        const docgen = b.addExecutable(.{ .name = "docgen", .root_module = b.createModule(.{ .root_source_file = b.path("tools/docgen/main.zig"), .target = target, .optimize = optimize, .link_libc = true }) });
        const run_docgen = b.addRunArtifact(docgen);
        if (b.args) |args| run_docgen.addArgs(args);
        b.step("docs-site", "Generate documentation website").dependOn(&run_docgen.step);
    }

    // Profile build
    if (pathExists("tools/cli/main.zig")) {
        var profile_opts = options;
        profile_opts.enable_profiling = true;
        const abi_profile = createAbiModule(b, profile_opts, target, optimize);
        const profile_exe = b.addExecutable(.{ .name = "abi-profile", .root_module = b.createModule(.{ .root_source_file = b.path("tools/cli/main.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true }) });
        profile_exe.root_module.addImport("abi", abi_profile);
        profile_exe.root_module.addImport("cli", createCliModule(b, abi_profile, target, optimize));
        b.installArtifact(profile_exe);
        b.step("profile", "Build with performance profiling").dependOn(b.getInstallStep());
    }

    // WASM
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
}
