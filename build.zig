const std = @import("std");

pub fn build(b: *std.Build) void {
    // ─── Standard build configuration ────────────────────────────────────────
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ─── Feature flags for conditional compilation ───────────────────────────
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
        .name = "zvim",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = platform_optimize,
    });

    // ─── Optimization flags ──────────────────────────────────────────────────
    exe.link_function_sections = true;
    exe.link_gc_sections = true;
    if (platform_optimize == .ReleaseSmall or platform_optimize == .ReleaseFast) {
        exe.strip = true;
    }

    // ─── Dependencies ────────────────────────────────────────────────────────
    exe.root_module.addImport("zli", b.dependency("zli", .{}).module("root"));
    exe.root_module.addImport("zf", b.dependency("zf", .{}).module("root"));
    exe.root_module.addImport("json", b.dependency("json", .{}).module("json"));
    exe.root_module.addImport("prompter", b.dependency("prompter", .{}).module("prompter"));
    exe.root_module.addOptions("build_options", options);

    // ─── Platform-specific dependencies ──────────────────────────────────────
    switch (target.result.os.tag) {
        .linux => {
            exe.linkSystemLibrary("c");
            if (b.option(bool, "enable_io_uring", "Enable io_uring support") orelse true) {
                exe.linkSystemLibrary("uring");
            }
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

    // ─── Build steps ─────────────────────────────────────────────────────────
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

    // ─── Cross-platform targets ──────────────────────────────────────────────
    addCrossTargets(b, exe, options);
}

fn addCrossTargets(b: *std.Build, exe: *std.Build.Step.Compile, options: *std.Build.Step.Options) void {
    const targets = [_]struct { name: []const u8, query: std.Target.Query }{
        .{ .name = "x86_64-linux", .query = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl } },
        .{ .name = "aarch64-linux", .query = .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu } },
        .{ .name = "x86_64-windows", .query = .{ .cpu_arch = .x86_64, .os_tag = .windows } },
        .{ .name = "x86_64-macos", .query = .{ .cpu_arch = .x86_64, .os_tag = .macos } },
        .{ .name = "aarch64-macos", .query = .{ .cpu_arch = .aarch64, .os_tag = .macos } },
        .{ .name = "aarch64-ios", .query = .{ .cpu_arch = .aarch64, .os_tag = .ios } },
    };

    const cross_step = b.step("cross", "Build for all supported platforms");

    for (targets) |t| {
        const cross_exe = b.addExecutable(.{
            .name = b.fmt("zvim-{s}", .{t.name}),
            .root_source_file = exe.root_source_file,
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
}
