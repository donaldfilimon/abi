const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU features") orelse false;
    const enable_web = b.option(bool, "enable-web", "Enable Web/WebGPU") orelse false;
    const enable_mon = b.option(bool, "enable-monitoring", "Enable metrics") orelse false;

    const abi_mod = b.addModule("abi", .{ .root_source_file = .{ .path = "src/mod.zig" } });
    abi_mod.addOptions("build_options", b.addOptions("build_options"));

    // Main executable: comprehensive CLI
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_source_file = .{ .path = "src/comprehensive_cli.zig" },
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("abi", abi_mod);

    if (enable_gpu) {
        // Add GPU deps
        _ = b.addLinkSystemLibrary("vulkan");
    }
    if (enable_web) {
        // Add WebGPU deps (placeholder)
    }
    if (enable_mon) {
        // Add monitoring libs
    }

    b.installArtifact(exe);

    // Unit test step
    const unit = b.addTest(.{ .root_source_file = .{ .path = "src/mod.zig" }, .target = target, .optimize = optimize });
    unit.root_module.addImport("abi", abi_mod);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&unit.step);

    // Benchmark step
    const bench = b.addTest(.{ .root_source_file = .{ .path = "benchmarks/main.zig" }, .target = target, .optimize = optimize });
    bench.root_module.addImport("abi", abi_mod);
    const bench_step = b.step("bench", "Run benchmark suite");
    bench_step.dependOn(&bench.step);

    // Docs step – placeholder using zig docgen
    const docs_step = b.step("docs", "Generate docs");
    _ = docs_step;
}
