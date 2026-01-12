const std = @import("std");
const builtin = @import("builtin");

fn pathExists(path: []const u8) bool {
    // For files in the project root, assume they exist
    // The actual compilation will fail if the file doesn't exist
    _ = path;
    return true;
}

const BuildOptions = struct {
    enable_gpu: bool,
    enable_ai: bool,
    enable_explore: bool,
    enable_llm: bool,
    enable_web: bool,
    enable_database: bool,
    enable_network: bool,
    enable_profiling: bool,
    gpu_cuda: bool,
    gpu_vulkan: bool,
    gpu_stdgpu: bool,
    gpu_metal: bool,
    gpu_webgpu: bool,
    gpu_opengl: bool,
    gpu_opengles: bool,
    gpu_webgl2: bool,
    cache_dir: []const u8,
    global_cache_dir: ?[]const u8,
};

const Defaults = struct {
    const enable_gpu = true;
    const enable_ai = true;
    const enable_explore = true;
    const enable_llm = true;
    const enable_web = true;
    const enable_database = true;
    const enable_network = true;
    const enable_profiling = true;
};

fn readBuildOptions(b: *std.Build) BuildOptions {
    const enable_gpu =
        b.option(bool, "enable-gpu", "Enable GPU support") orelse
        Defaults.enable_gpu;
    const enable_ai =
        b.option(bool, "enable-ai", "Enable AI features") orelse
        Defaults.enable_ai;
    const enable_explore =
        b.option(bool, "enable-explore", "Enable AI code exploration") orelse
        (enable_ai and Defaults.enable_explore);
    const enable_llm =
        b.option(bool, "enable-llm", "Enable local LLM inference") orelse
        (enable_ai and Defaults.enable_llm);
    const enable_web =
        b.option(bool, "enable-web", "Enable web features") orelse
        Defaults.enable_web;
    const enable_database =
        b.option(bool, "enable-database", "Enable database features") orelse
        Defaults.enable_database;
    const enable_network =
        b.option(bool, "enable-network", "Enable network distributed compute") orelse
        Defaults.enable_network;
    const enable_profiling =
        b.option(bool, "enable-profiling", "Enable profiling and metrics") orelse
        Defaults.enable_profiling;

    const cache_dir =
        b.option([]const u8, "cache-dir", "Directory for build cache") orelse
        ".zig-cache";

    const global_cache_dir =
        b.option([]const u8, "global-cache-dir", "Directory for global build cache") orelse
        null;

    // GPU backend options - only enable Vulkan by default for cross-platform compatibility
    const gpu_cuda = b.option(bool, "gpu-cuda", "Enable CUDA GPU backend") orelse false;
    const gpu_vulkan = b.option(bool, "gpu-vulkan", "Enable Vulkan GPU backend") orelse enable_gpu;
    const gpu_stdgpu = b.option(bool, "gpu-stdgpu", "Enable Zig std.gpu SPIR-V backend") orelse false;
    const gpu_metal = b.option(bool, "gpu-metal", "Enable Metal GPU backend") orelse false;
    const gpu_webgpu = b.option(bool, "gpu-webgpu", "Enable WebGPU backend") orelse enable_web;
    const gpu_opengl = b.option(bool, "gpu-opengl", "Enable OpenGL backend") orelse false;
    const gpu_opengles =
        b.option(bool, "gpu-opengles", "Enable OpenGL ES backend") orelse false;
    const gpu_webgl2 = b.option(bool, "gpu-webgl2", "Enable WebGL2 backend") orelse enable_web;

    // Validate GPU backend combinations
    if (gpu_cuda and gpu_vulkan) {
        std.log.warn("Both CUDA and Vulkan backends enabled; this may cause conflicts. Consider using only one GPU backend.", .{});
    }
    if (gpu_opengl and gpu_webgl2) {
        std.log.warn("Both OpenGL and WebGL2 backends enabled; prefer one or the other.", .{});
    }

    return .{
        .enable_gpu = enable_gpu,
        .enable_ai = enable_ai,
        .enable_explore = enable_explore,
        .enable_llm = enable_llm,
        .enable_web = enable_web,
        .enable_database = enable_database,
        .enable_network = enable_network,
        .enable_profiling = enable_profiling,
        .gpu_cuda = gpu_cuda,
        .gpu_vulkan = gpu_vulkan,
        .gpu_stdgpu = gpu_stdgpu,
        .gpu_metal = gpu_metal,
        .gpu_webgpu = gpu_webgpu,
        .gpu_opengl = gpu_opengl,
        .gpu_opengles = gpu_opengles,
        .gpu_webgl2 = gpu_webgl2,
        .cache_dir = cache_dir,
        .global_cache_dir = global_cache_dir,
    };
}

/// Create build options module with centralized defaults.
fn createBuildOptionsModule(b: *std.Build, options: BuildOptions) *std.Build.Module {
    var build_options = b.addOptions();

    // Package version
    build_options.addOption([]const u8, "package_version", "0.1.0");

    build_options.addOption(bool, "enable_gpu", options.enable_gpu);
    build_options.addOption(bool, "enable_ai", options.enable_ai);
    build_options.addOption(bool, "enable_explore", options.enable_explore);
    build_options.addOption(bool, "enable_llm", options.enable_llm);
    build_options.addOption(bool, "enable_web", options.enable_web);
    build_options.addOption(bool, "enable_database", options.enable_database);
    build_options.addOption(bool, "enable_network", options.enable_network);
    build_options.addOption(bool, "enable_profiling", options.enable_profiling);

    // GPU backend options
    build_options.addOption(bool, "gpu_cuda", options.gpu_cuda);
    build_options.addOption(bool, "gpu_vulkan", options.gpu_vulkan);
    build_options.addOption(bool, "gpu_stdgpu", options.gpu_stdgpu);
    build_options.addOption(bool, "gpu_metal", options.gpu_metal);
    build_options.addOption(bool, "gpu_webgpu", options.gpu_webgpu);
    build_options.addOption(bool, "gpu_opengl", options.gpu_opengl);
    build_options.addOption(bool, "gpu_opengles", options.gpu_opengles);
    build_options.addOption(bool, "gpu_webgl2", options.gpu_webgl2);

    return build_options.createModule();
}

fn createCliModule(
    b: *std.Build,
    abi_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const cli_module = b.createModule(.{
        .root_source_file = b.path("tools/cli/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli_module.addImport("abi", abi_module);
    return cli_module;
}

fn warnInconsistentOptions(options: BuildOptions) void {
    if (!options.enable_gpu and
        (options.gpu_cuda or options.gpu_vulkan or options.gpu_stdgpu or options.gpu_metal or
            options.gpu_opengl or options.gpu_opengles))
    {
        std.log.warn(
            "GPU backends enabled but enable-gpu=false; " ++
                "backends will be inactive until GPU is enabled",
            .{},
        );
    }
    if (!options.enable_web and (options.gpu_webgpu or options.gpu_webgl2)) {
        std.log.warn(
            "Web GPU backends enabled but enable-web=false; web GPU backends will be inactive",
            .{},
        );
    }
}

fn validateFeatureFlags(options: BuildOptions) !void {
    const invalid_combos = [2]struct { enabled: bool, required: bool, name: []const u8 }{
        .{
            .enabled = options.gpu_cuda or options.gpu_vulkan or options.gpu_stdgpu or options.gpu_metal,
            .required = options.enable_gpu,
            .name = "enable-gpu",
        },
        .{
            .enabled = options.gpu_webgpu or options.gpu_webgl2,
            .required = options.enable_web,
            .name = "enable-web",
        },
    };

    for (invalid_combos) |combo| {
        if (combo.enabled and !combo.required) {
            std.log.err(
                "Feature flag validation failed: GPU backend enabled without {s}=true",
                .{combo.name},
            );
            std.log.err("Enable {s} or disable GPU backend flags", .{combo.name});
            return error.InvalidFeatureCombination;
        }
    }

    if (options.gpu_webgpu and options.gpu_cuda) {
        std.log.warn(
            "Both WebGPU and CUDA backends enabled; this is unusual configuration",
            .{},
        );
    }

    if (options.gpu_webgl2 and options.gpu_opengl) {
        std.log.warn(
            "Both WebGL2 and OpenGL backends enabled; prefer one or the other",
            .{},
        );
    }
}

// Determine if a file exists using std.fs. The original implementation used
// std.Io, but that requires a correctly initialised `IO` instance which is
// fragile across Zig versions. The logic below simply probes the filesystem
// using the current working directory.

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The original build script used a `std.Io.Threaded` instance. Replacing
    // it with the single‑threaded `std.Io` keeps the behaviour consistent
    // while avoiding the complex initialization required by Threaded.
    // `io` is no longer needed.

    // Create build options module
    const base_options = readBuildOptions(b);
    warnInconsistentOptions(base_options);
    validateFeatureFlags(base_options) catch |err| {
        std.log.err("Feature flag validation failed: {t}", .{err});
        std.process.exit(1);
    };
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
        const cli_module = createCliModule(b, abi_module, target, optimize);
        exe.root_module.addImport("abi", abi_module);
        exe.root_module.addImport("cli", cli_module);
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
        // Fall back to src/main.zig as the CLI entrypoint
        const exe = b.addExecutable(.{
            .name = "abi",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/main.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("abi", abi_module);

        b.installArtifact(exe);

        // Run step for CLI fallback
        const run_cli = b.addRunArtifact(exe);
        run_cli.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_cli.addArgs(args);
        }

        const run_step = b.step("run", "Run the ABI CLI");
        run_step.dependOn(&run_cli.step);
    }

    // Example programs
    const examples_step = b.step("examples", "Build all examples");
    const example_names = [_][]const u8{
        "hello",
        "database",
        "agent",
        "compute",
        "gpu",
        "network",
        "discord",
    };

    for (example_names) |example_name| {
        const example_path = b.fmt("examples/{s}.zig", .{example_name});
        if (pathExists(example_path)) {
            const example_exe = b.addExecutable(.{
                .name = b.fmt("example-{s}", .{example_name}),
                .root_module = b.createModule(.{
                    .root_source_file = b.path(example_path),
                    .target = target,
                    .optimize = optimize,
                }),
            });
            example_exe.root_module.addImport("abi", abi_module);
            b.installArtifact(example_exe);

            const run_example = b.addRunArtifact(example_exe);
            if (b.args) |args| {
                run_example.addArgs(args);
            }

            const example_run_step = b.step(b.fmt("run-{s}", .{example_name}), b.fmt("Run {s} example", .{example_name}));
            example_run_step.dependOn(b.getInstallStep());
            example_run_step.dependOn(&run_example.step);

            examples_step.dependOn(&example_exe.step);
        }
    }

    // Test suite
    const has_tests = pathExists("src/tests/mod.zig");
    if (has_tests) {
        const main_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/tests/mod.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        main_tests.root_module.addImport("abi", abi_module);
        main_tests.root_module.addImport("build_options", build_options_module);

        const run_main_tests = b.addRunArtifact(main_tests);
        run_main_tests.skip_foreign_checks = true;

        const test_step = b.step("test", "Run unit tests");
        test_step.dependOn(&run_main_tests.step);
    } else {
        std.log.warn("src/tests/mod.zig not found; skipping test step", .{});
    }

    // Benchmark step
    const has_benchmark = pathExists("src/compute/runtime/benchmark.zig");
    if (has_benchmark) {
        const benchmark_exe = b.addExecutable(.{
            .name = "abi-benchmark",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/compute/runtime/benchmark.zig"),
                .target = target,
                .optimize = .ReleaseFast,
            }),
        });
        benchmark_exe.root_module.addImport("abi", abi_module);

        const run_benchmark = b.addRunArtifact(benchmark_exe);

        const benchmark_step = b.step("benchmark-legacy", "Run legacy performance benchmarks");
        benchmark_step.dependOn(&run_benchmark.step);
    } else {
        std.log.warn("src/compute/runtime/benchmark.zig not found; skipping benchmark step", .{});
    }

    // Benchmarks
    if (pathExists("benchmarks/run.zig")) {
        const benchmark_exe = b.addExecutable(.{
            .name = "benchmarks",
            .root_module = b.createModule(.{
                .root_source_file = b.path("benchmarks/run.zig"),
                .target = target,
                .optimize = .ReleaseFast,
            }),
        });
        benchmark_exe.root_module.addImport("abi", abi_module);

        const run_benchmarks = b.addRunArtifact(benchmark_exe);

        const benchmarks_step = b.step("benchmarks", "Run comprehensive benchmarks");
        benchmarks_step.dependOn(&run_benchmarks.step);
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
        const cli_profile_module = createCliModule(b, abi_profile_module, target, optimize);
        profile_exe.root_module.addImport("abi", abi_profile_module);
        profile_exe.root_module.addImport("cli", cli_profile_module);
        b.installArtifact(profile_exe);

        const profile_step = b.step("profile", "Build with performance profiling");
        profile_step.dependOn(b.getInstallStep());
    }

    // WASM Build
    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    var wasm_options = base_options;
    // WASM environment constraints
    wasm_options.enable_database = false; // No std.Io.Threaded
    wasm_options.enable_network = false; // No socket support
    wasm_options.enable_gpu = false; // Explicitly disable GPU defaults
    wasm_options.gpu_cuda = false;
    wasm_options.gpu_vulkan = false;
    wasm_options.gpu_metal = false;
    wasm_options.gpu_opengl = false;
    wasm_options.gpu_opengles = false;

    // WebGPU can technically work via bindings, but let's disable to simplify first pass
    wasm_options.enable_web = false;

    // Create a specific module for WASM that uses these restricted options
    const wasm_build_options_module = createBuildOptionsModule(b, wasm_options);
    const abi_wasm_module = b.addModule("abi-wasm", .{
        .root_source_file = b.path("src/abi.zig"),
        .target = wasm_target,
        .optimize = optimize,
    });
    abi_wasm_module.addImport("build_options", wasm_build_options_module);

    const wasm_lib = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bindings/wasm/abi_wasm.zig"),
            .target = wasm_target,
            .optimize = optimize,
        }),
    });
    wasm_lib.entry = .disabled;
    wasm_lib.rdynamic = true;
    wasm_lib.root_module.addImport("abi", abi_wasm_module);

    const check_wasm = b.step("check-wasm", "Check WASM compilation");
    check_wasm.dependOn(&wasm_lib.step);

    const install_wasm = b.addInstallArtifact(wasm_lib, .{
        .dest_dir = .{ .override = .{ .custom = "wasm" } },
    });
    const wasm_step = b.step("wasm", "Build WASM bindings");
    wasm_step.dependOn(&install_wasm.step);
}
