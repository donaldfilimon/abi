const std = @import("std");

/// Create build options module with centralized defaults
fn createBuildOptions(b: *std.Build) *std.Build.Module {
    const build_options = b.addOptions();

    // Package version
    build_options.addOption([]const u8, "package_version", "0.2.0");

    // Feature flags - all default to true for full functionality
    const enable_gpu = b.option(bool, "enable-gpu", "Enable GPU support") orelse true;
    const enable_ai = b.option(bool, "enable-ai", "Enable AI features") orelse true;
    const enable_web = b.option(bool, "enable-web", "Enable web features") orelse true;
    const enable_database = b.option(bool, "enable-database", "Enable database features") orelse true;

    build_options.addOption(bool, "enable_gpu", enable_gpu);
    build_options.addOption(bool, "enable_ai", enable_ai);
    build_options.addOption(bool, "enable_web", enable_web);
    build_options.addOption(bool, "enable_database", enable_database);

    return build_options.createModule();
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create build options module
    const build_options_module = createBuildOptions(b);

    // Accelerator module
    const accelerator_module = b.addModule("accelerator", .{
        .root_source_file = b.path("lib/accelerator/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    accelerator_module.addImport("build_options", build_options_module);

    // Core library module
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("lib/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_options_module);
    abi_module.addImport("accelerator", accelerator_module);

    // CLI executable
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("abi", abi_module);
    b.installArtifact(exe);

    // Run step for CLI
    const run_cli = b.addRunArtifact(exe);
    run_cli.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cli.addArgs(args);
    }

    const run_step = b.step("run", "Run the ABI CLI");
    run_step.dependOn(&run_cli.step);

    // Neural Network Training Example
    const nn_training_exe = b.addExecutable(.{
        .name = "neural-network-training",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/neural_network_training.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    nn_training_exe.root_module.addImport("abi", abi_module);
    b.installArtifact(nn_training_exe);

    const run_nn_training = b.addRunArtifact(nn_training_exe);
    const run_nn_training_step = b.step("run-nn-training", "Run neural network training example");
    run_nn_training_step.dependOn(&run_nn_training.step);

    // Test suite
    const main_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    main_tests.root_module.addImport("abi", abi_module);

    const abi_tests = b.addTest(.{
        .root_module = abi_module,
    });

    const accelerator_tests = b.addTest(.{
        .root_module = accelerator_module,
    });

    const run_main_tests = b.addRunArtifact(main_tests);
    run_main_tests.skip_foreign_checks = true;
    const run_abi_tests = b.addRunArtifact(abi_tests);
    const run_accelerator_tests = b.addRunArtifact(accelerator_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_main_tests.step);
    test_step.dependOn(&run_abi_tests.step);
    test_step.dependOn(&run_accelerator_tests.step);
}
