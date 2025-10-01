const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const enable_gpu = b.option(bool, "enable_gpu", "Enable GPU feature") orelse false;
    const enable_web = b.option(bool, "enable_web", "Enable Web feature") orelse false;
    const enable_monitoring = b.option(bool, "enable_monitoring", "Enable monitoring feature") orelse true;

    const core_mod = b.createModule(.{ .root_source_file = b.path("src/core/mod.zig"), .target = target, .optimize = optimize });
    const core_lib = b.addStaticLibrary(.{ .name = "abi_core", .root_module = core_mod });

    const features = [_][]const u8{ "ai", "database", "gpu", "web", "monitoring", "connectors" };
    var feature_libs = std.ArrayList(*std.Build.Step.Compile).init(b.allocator);
    defer feature_libs.deinit();

    for (features) |feat| {
        if (std.mem.eql(u8, feat, "gpu") and !enable_gpu) continue;
        if (std.mem.eql(u8, feat, "web") and !enable_web) continue;
        if (std.mem.eql(u8, feat, "monitoring") and !enable_monitoring) continue;

        const mod_path = b.fmt("src/features/{s}/mod.zig", .{feat});
        const feat_mod = b.createModule(.{ .root_source_file = b.path(mod_path), .target = target, .optimize = optimize });
        const lib = b.addStaticLibrary(.{ .name = b.fmt("abi_{s}", .{feat}), .root_module = feat_mod });
        lib.linkLibrary(core_lib);
        feature_libs.append(lib) catch unreachable;
    }

    const cli_mod = b.createModule(.{ .root_source_file = b.path("src/cli/main.zig"), .target = target, .optimize = optimize });
    const cli_exe = b.addExecutable(.{ .name = "abi-cli", .root_module = cli_mod, .target = target, .optimize = optimize });
    for (feature_libs.items) |lib| cli_exe.linkLibrary(lib);
    cli_exe.linkLibrary(core_lib);
    b.installArtifact(cli_exe);

    const test_step = b.step("test", "Run unit tests for ABI features");

    const core_tests_mod = b.createModule(.{ .root_source_file = b.path("src/core/tests/mod.zig"), .target = target, .optimize = optimize });
    const core_tests = b.addTest(.{ .root_module = core_tests_mod });
    const core_run = b.addRunArtifact(core_tests);
    core_run.skip_foreign_checks = true;
    test_step.dependOn(&core_run.step);

    inline for (features) |feat| {
        if (std.mem.eql(u8, feat, "gpu") and !enable_gpu) continue;
        if (std.mem.eql(u8, feat, "web") and !enable_web) continue;
        if (std.mem.eql(u8, feat, "monitoring") and !enable_monitoring) continue;
        const test_mod_path = b.fmt("src/features/{s}/tests/mod.zig", .{feat});
        const tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path(test_mod_path), .target = target, .optimize = optimize }) });
        const run = b.addRunArtifact(tests);
        run.skip_foreign_checks = true;
        test_step.dependOn(&run.step);
    }

    const run_exe = b.addRunArtifact(cli_exe);
    const run = b.step("run", "Run the CLI (local)");
    run.dependOn(&run_exe.step);
}
