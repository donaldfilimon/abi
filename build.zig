const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

<<<<<<< HEAD
    // Main executable
    const exe = b.addExecutable(.{
        .name = "wdbx-ai",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add library dependencies
    const lib = b.addStaticLibrary(.{
        .name = "wdbx-ai-lib",
        .root_source_file = .{ .path = "src/mod.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Install artifacts
    b.installArtifact(exe);
    b.installArtifact(lib);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    // Test step
    const test_step = b.step("test", "Run unit tests");

    // Add test executables for each test file
    const test_files = [_][]const u8{
        "tests/test_ai.zig",
        "tests/test_database.zig",
        "tests/test_database_hnsw.zig",
        "tests/test_database_integration.zig",
        "tests/test_memory_management.zig",
        "tests/test_performance_optimizations.zig",
        "tests/test_simd_vector.zig",
        "tests/test_weather.zig",
        "tests/test_web_server.zig",
        "tests/test_cli_integration.zig",
        "tests/test_discord_plugin.zig",
        "tests/test_weather_integration.zig",
        "tests/test_performance_regression.zig",
    };

    for (test_files) |test_file| {
        const test_exe = b.addTest(.{
            .root_source_file = .{ .path = test_file },
            .target = target,
            .optimize = optimize,
        });

        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
=======
    // Feature flags for conditional compilation
    const options = b.addOptions();
    options.addOption(bool, "enable_gpu", b.option(bool, "gpu", "Enable GPU acceleration") orelse detectGPUSupport());
    options.addOption(bool, "enable_simd", b.option(bool, "simd", "Enable SIMD optimizations") orelse detectSIMDSupport());
    options.addOption(bool, "enable_tracy", b.option(bool, "tracy", "Enable Tracy profiler") orelse false);

    // Platform-specific optimizations
    const platform_optimize = switch (target.result.os.tag) {
        .ios => .ReleaseSmall,
        .windows => .ReleaseSafe,
        else => optimize,
    };

    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = platform_optimize,
    });

    // Optimization flags
    exe.link_function_sections = true;
    exe.link_gc_sections = true;
    if (platform_optimize == .ReleaseSmall or platform_optimize == .ReleaseFast) {
        exe.root_module.strip = true;
>>>>>>> 850019f80c86681d50ab87479a8796ce3e849dda
    }

    // Dependencies
    exe.root_module.addOptions("build_options", options);

<<<<<<< HEAD
    // Main module tests
    const main_test = b.addTest(.{
        .root_source_file = .{ .path = "src/mod.zig" },
        .target = target,
        .optimize = optimize,
    });
    const run_main_test = b.addRunArtifact(main_test);
    test_step.dependOn(&run_main_test.step);

    // Benchmark step
    const benchmark_step = b.step("benchmark", "Run benchmarks");

    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = .{ .path = "benchmarks/main.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });

    const run_benchmark = b.addRunArtifact(benchmark_exe);
    benchmark_step.dependOn(&run_benchmark.step);

    // Documentation step
    const docs_step = b.step("docs", "Generate documentation");
    const docs_install = b.addInstallDirectory(.{
        .source_dir = lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&docs_install.step);

    // Format step
    const fmt_step = b.step("fmt", "Format source code");
    const fmt = b.addFmt(.{
        .paths = &.{
            "src",
            "tests",
            "benchmarks",
            "build.zig",
=======
    // Platform-specific dependencies
    switch (target.result.os.tag) {
        .linux => {
            exe.linkSystemLibrary("c");
            if (b.option(bool, "enable_io_uring", "Enable io_uring support") orelse false) {
                exe.linkSystemLibrary("uring");
            }
>>>>>>> 850019f80c86681d50ab87479a8796ce3e849dda
        },
        .windows => {
            exe.linkSystemLibrary("kernel32");
            exe.linkSystemLibrary("user32");
            exe.linkSystemLibrary("d3d12");
        },
        .macos, .ios => {
            exe.linkFramework("Metal");
            exe.linkFramework("MetalKit");
            exe.linkFramework("CoreGraphics");
        },
        else => {},
    }

    b.installArtifact(exe);

    const bench_step = b.step("bench", "Run performance benchmarks");
    const bench_exe = b.addRunArtifact(exe);
    bench_exe.addArg("bench");
    bench_exe.addArg("--iterations=1000");
    bench_step.dependOn(&bench_exe.step);

    const test_step = b.step("test", "Run unit tests");
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = platform_optimize,
    });
<<<<<<< HEAD
    dev_exe.addArgs(&.{"-DDEBUG"});

    const dev_step = b.step("dev", "Build development version with debug symbols");
    dev_step.dependOn(&b.addInstallArtifact(dev_exe, .{}).step);

    // Production build
    const prod_exe = b.addExecutable(.{
        .name = "wdbx-ai-prod",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    prod_exe.strip = true;

    const prod_step = b.step("prod", "Build optimized production version");
    prod_step.dependOn(&b.addInstallArtifact(prod_exe, .{}).step);
=======
    unit_tests.root_module.addOptions("build_options", options);
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);

    addCrossTargets(b, exe, options);
}
>>>>>>> 850019f80c86681d50ab87479a8796ce3e849dda

fn addCrossTargets(b: *std.Build, exe: *std.Build.Step.Compile, options: *std.Build.Step.Options) void {
    const targets = [_]struct { name: []const u8, query: std.Target.Query }{
        .{ .name = "x86_64-linux", .query = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl } },
        .{ .name = "aarch64-linux", .query = .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu } },
        .{ .name = "x86_64-windows", .query = .{ .cpu_arch = .x86_64, .os_tag = .windows } },
        .{ .name = "x86_64-macos", .query = .{ .cpu_arch = .x86_64, .os_tag = .macos } },
        .{ .name = "aarch64-macos", .query = .{ .cpu_arch = .aarch64, .os_tag = .macos } },
        .{ .name = "aarch64-ios", .query = .{ .cpu_arch = .aarch64, .os_tag = .ios } },
    };

<<<<<<< HEAD
    // All step - runs everything
    const all_step = b.step("all", "Build everything and run all checks");
    all_step.dependOn(check_step);
    all_step.dependOn(benchmark_step);
    all_step.dependOn(docs_step);
    all_step.dependOn(&b.getInstallStep().step);
=======
    const cross_step = b.step("cross", "Build for all supported platforms");

    for (targets) |t| {
        const cross_exe = b.addExecutable(.{
            .name = b.fmt("zvim-{s}", .{t.name}),
            .root_source_file = b.path("src/main.zig"),
            .target = b.resolveTargetQuery(t.query),
            .optimize = exe.root_module.optimize orelse .ReleaseSafe,
        });

        cross_exe.root_module.addOptions("build_options", options);
        const install = b.addInstallArtifact(cross_exe, .{});
        cross_step.dependOn(&install.step);
    }
}

fn detectGPUSupport() bool {
    return true;
}

fn detectSIMDSupport() bool {
    return switch (builtin.cpu.arch) {
        .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .avx2),
        .aarch64 => std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon),
        else => false,
    };
>>>>>>> 850019f80c86681d50ab87479a8796ce3e849dda
}
