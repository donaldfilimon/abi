const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Feature Flags
    const feat_ai = b.option(bool, "feat-ai", "Enable AI features") orelse true;
    const feat_gpu = b.option(bool, "feat-gpu", "Enable GPU acceleration") orelse true;

    const options = b.addOptions();
    options.addOption(bool, "feat_ai", feat_ai);
    options.addOption(bool, "feat_gpu", feat_gpu);
    const options_mod = options.createModule();

    // ABI Module
    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_mod.addImport("build_options", options_mod);

    // CLI Executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    b.installArtifact(exe);

    // Steps
    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const cli_step = b.step("cli", "Build ABI CLI");
    cli_step.dependOn(b.getInstallStep());

    // Tests
    const mod_tests = b.addTest(.{
        .root_module = abi_mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);

    const check_step = b.step("check", "Run all checks (tests + lint)");
    check_step.dependOn(test_step);
}
