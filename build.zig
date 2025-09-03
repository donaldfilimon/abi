const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable
    const exe = b.addExecutable(.{
        .name = "wdbx-ai",
        .root_source_file = .{ .path = "src/main.zig" },
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
    }

    // Core module tests
    const core_test = b.addTest(.{
        .root_source_file = .{ .path = "src/core/mod.zig" },
        .target = target,
        .optimize = optimize,
    });
    const run_core_test = b.addRunArtifact(core_test);
    test_step.dependOn(&run_core_test.step);

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
        },
    });
    fmt_step.dependOn(&fmt.step);

    // Check step (format + test)
    const check_step = b.step("check", "Check code formatting and run tests");
    check_step.dependOn(&fmt.step);
    check_step.dependOn(test_step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    const clean_cmd = b.addSystemCommand(&.{ "rm", "-rf", "zig-cache", "zig-out" });
    clean_step.dependOn(&clean_cmd.step);

    // Install step for CLI tool
    const install_step = b.step("install", "Install WDBX-AI CLI tool");
    const install_exe = b.addInstallArtifact(exe, .{
        .dest_dir = .{ .override = .{ .custom = "../bin" } },
    });
    install_step.dependOn(&install_exe.step);

    // Development mode with debug symbols
    const dev_exe = b.addExecutable(.{
        .name = "wdbx-ai-dev",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = .Debug,
    });
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

    // Static analysis step
    const analyze_step = b.step("analyze", "Run static analysis");
    const analyze_cmd = b.addSystemCommand(&.{ "zig", "ast-check", "src/main.zig" });
    analyze_step.dependOn(&analyze_cmd.step);

    // All step - runs everything
    const all_step = b.step("all", "Build everything and run all checks");
    all_step.dependOn(check_step);
    all_step.dependOn(benchmark_step);
    all_step.dependOn(docs_step);
    all_step.dependOn(&b.getInstallStep().step);
}