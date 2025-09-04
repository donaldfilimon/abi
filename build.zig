const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options for conditional compilation
    const options = b.addOptions();
    options.addOption(bool, "enable_gpu", b.option(bool, "gpu", "Enable GPU acceleration") orelse true);
    options.addOption(bool, "enable_simd", b.option(bool, "simd", "Enable SIMD optimizations") orelse true);

    const exe = b.addExecutable(.{
        .name = "zvim",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Dependencies
    exe.root_module.addOptions("build_options", options);

    // Link system libraries
    exe.linkSystemLibrary("c");

    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const bench_step = b.step("bench", "Run performance benchmarks");
    const bench_exe = b.addRunArtifact(exe);
    bench_exe.addArg("bench");
    bench_exe.addArg("--iterations=1000");
    bench_step.dependOn(&bench_exe.step);

    const test_step = b.step("test", "Run unit tests");
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    unit_tests.root_module.addOptions("build_options", options);
    test_step.dependOn(&b.addRunArtifact(unit_tests).step);
}
