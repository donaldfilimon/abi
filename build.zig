const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library
    const lib = b.addStaticLibrary(.{
        .name = "wdbx",
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Main executable
    const exe = b.addExecutable(.{
        .name = "wdbx",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe);

    // CLI executable
    const cli = b.addExecutable(.{
        .name = "wdbx-cli",
        .root_source_file = b.path("src/wdbx/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(cli);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Test step
    const test_step = b.step("test", "Run unit tests");
    
    // Core module tests
    const core_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_core_tests = b.addRunArtifact(core_unit_tests);
    test_step.dependOn(&run_core_tests.step);
    
    // Test runner executable
    const test_runner = b.addExecutable(.{
        .name = "test_runner",
        .root_source_file = b.path("tests/run_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add module path so tests can import wdbx
    test_runner.root_module.addImport("wdbx", &lib.root_module);
    
    const run_test_runner = b.addRunArtifact(test_runner);
    const test_runner_step = b.step("test-all", "Run all tests with test runner");
    test_runner_step.dependOn(&run_test_runner.step);
    
    // Individual test suites
    const test_suites = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "test-core", .path = "tests/core_tests.zig" },
        .{ .name = "test-database", .path = "tests/database_tests.zig" },
        .{ .name = "test-integration", .path = "tests/integration_tests.zig" },
    };
    
    for (test_suites) |suite| {
        const suite_test = b.addTest(.{
            .root_source_file = b.path(suite.path),
            .target = target,
            .optimize = optimize,
        });
        suite_test.root_module.addImport("wdbx", &lib.root_module);
        
        const run_suite = b.addRunArtifact(suite_test);
        const suite_step = b.step(suite.name, b.fmt("Run {s}", .{suite.name}));
        suite_step.dependOn(&run_suite.step);
    }

    // Benchmark executable
    const bench = b.addExecutable(.{
        .name = "wdbx-bench",
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    b.installArtifact(bench);

    const bench_cmd = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench_cmd.step);

    // Documentation generation
    const docs_step = b.step("docs", "Generate documentation");
    const docs = b.addStaticLibrary(.{
        .name = "wdbx-docs",
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
        .optimize = .Debug,
    });
    docs.root_module.emit_docs = .emit;
    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&install_docs.step);

    // Format check
    const fmt_step = b.step("fmt", "Format all source files");
    const fmt = b.addFmt(.{
        .paths = &.{ "src", "tests", "benchmarks", "examples" },
        .check = false,
    });
    fmt_step.dependOn(&fmt.step);

    const fmt_check_step = b.step("fmt-check", "Check source formatting");
    const fmt_check = b.addFmt(.{
        .paths = &.{ "src", "tests", "benchmarks", "examples" },
        .check = true,
    });
    fmt_check_step.dependOn(&fmt_check.step);
}