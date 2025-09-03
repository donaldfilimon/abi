//! Build configuration for refactored WDBX Vector Database

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable
    const exe = b.addExecutable(.{
        .name = "wdbx",
        .root_source_file = b.path("src/main_refactored.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Core module
    const core_module = b.addModule("core", .{
        .root_source_file = b.path("src/core/mod.zig"),
    });

    // API module
    const api_module = b.addModule("api", .{
        .root_source_file = b.path("src/api/mod.zig"),
    });

    // Utils module
    const utils_module = b.addModule("utils", .{
        .root_source_file = b.path("src/utils/mod.zig"),
    });

    // Add dependencies
    exe.root_module.addImport("core", core_module);
    exe.root_module.addImport("api", api_module);
    exe.root_module.addImport("utils", utils_module);

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    // Test configuration
    const test_filters = [_][]const u8{
        "src/main_refactored.zig",
        "src/core/database.zig",
        "src/core/vector/mod.zig",
        "src/core/vector/distance.zig",
        "src/core/vector/simd.zig",
        "src/core/index/mod.zig",
        "src/core/index/flat.zig",
        "src/core/index/hnsw.zig",
        "src/core/storage/mod.zig",
        "src/core/storage/file.zig",
        "src/core/storage/memory.zig",
        "src/utils/errors.zig",
    };

    const test_step = b.step("test", "Run unit tests");
    for (test_filters) |filter| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path(filter),
            .target = target,
            .optimize = optimize,
        });
        
        unit_tests.root_module.addImport("core", core_module);
        unit_tests.root_module.addImport("api", api_module);
        unit_tests.root_module.addImport("utils", utils_module);
        
        const run_unit_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_unit_tests.step);
    }

    // Benchmark executable
    const benchmark_exe = b.addExecutable(.{
        .name = "wdbx-benchmark",
        .root_source_file = b.path("benchmarks/performance_suite.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    
    benchmark_exe.root_module.addImport("core", core_module);
    b.installArtifact(benchmark_exe);

    const benchmark_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_cmd.step.dependOn(b.getInstallStep());
    const benchmark_step = b.step("benchmark", "Run performance benchmarks");
    benchmark_step.dependOn(&benchmark_cmd.step);

    // Documentation generation
    const docs_step = b.step("docs", "Generate documentation");
    const docs_exe = b.addExecutable(.{
        .name = "wdbx-docs",
        .root_source_file = b.path("src/main_refactored.zig"),
        .target = target,
        .optimize = .Debug,
    });
    const docs_cmd = b.addRunArtifact(docs_exe);
    docs_cmd.addArg("--emit-docs");
    docs_step.dependOn(&docs_cmd.step);

    // Static analysis
    const analyze_step = b.step("analyze", "Run static analysis");
    const analyze_exe = b.addExecutable(.{
        .name = "static-analysis",
        .root_source_file = b.path("tools/static_analysis.zig"),
        .target = target,
        .optimize = optimize,
    });
    analyze_exe.root_module.addImport("core", core_module);
    const analyze_cmd = b.addRunArtifact(analyze_exe);
    analyze_step.dependOn(&analyze_cmd.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    clean_step.dependOn(&b.addRemoveDirTree("zig-out").step);
    clean_step.dependOn(&b.addRemoveDirTree("zig-cache").step);

    // Format step
    const fmt_step = b.step("fmt", "Format source code");
    const fmt = b.addFmt(.{
        .paths = &.{ "src", "tests", "benchmarks", "tools" },
    });
    fmt_step.dependOn(&fmt.step);

    // Check step (format check)
    const check_step = b.step("check", "Check code formatting");
    const fmt_check = b.addFmt(.{
        .paths = &.{ "src", "tests", "benchmarks", "tools" },
        .check = true,
    });
    check_step.dependOn(&fmt_check.step);
}