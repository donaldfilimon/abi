const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main module
    const main_mod = b.addModule("wdbx-ai", .{
        .root_source_file = b.path("src/mod.zig"),
        .target = target,
    });

    // Main executable using the simple main.zig
    const exe = b.addExecutable(.{
        .name = "wdbx-ai",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "wdbx-ai", .module = main_mod },
            },
        }),
    });

    // Install artifacts
    b.installArtifact(exe);

    // Run command
    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Test step
    const test_step = b.step("test", "Run tests");
    
    // Module tests
    const mod_tests = b.addTest(.{
        .root_module = main_mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    test_step.dependOn(&run_mod_tests.step);

    // Format step
    const fmt_step = b.step("fmt", "Format source code");
    const fmt = b.addFmt(.{
        .paths = &.{
            "src",
            "build.zig",
        },
    });
    fmt_step.dependOn(&fmt.step);

    // Static analysis step
    const analyze_step = b.step("analyze", "Run static analysis");
    const analyze_cmd = b.addSystemCommand(&.{ "zig", "ast-check", "src/main.zig" });
    analyze_step.dependOn(&analyze_cmd.step);
}