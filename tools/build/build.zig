const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Feature Flags
    const feat_ai = b.option(bool, "feat-ai", "Enable AI features") orelse true;
    const feat_gpu = b.option(bool, "feat-gpu", "Enable GPU acceleration") orelse true;
    const feat_tui = b.option(bool, "feat-tui", "Enable TUI features") orelse false;
    const feat_accelerator = b.option(bool, "feat-accelerator", "Enable accelerator backend selection") orelse true;
    const feat_shader = b.option(bool, "feat-shader", "Enable Zig shader validation backend") orelse true;
    const feat_mlir = b.option(bool, "feat-mlir", "Enable textual MLIR lowering backend") orelse true;
    const feat_mobile = b.option(bool, "feat-mobile", "Enable mobile platform feature flag") orelse false;
    const feat_wdbx = b.option(bool, "feat-wdbx", "Enable WDBX vector store and block memory") orelse true;
    const feat_os_control = b.option(bool, "feat-os-control", "Enable OS command policy controls") orelse true;

    const options = b.addOptions();
    options.addOption(bool, "feat_ai", feat_ai);
    options.addOption(bool, "feat_gpu", feat_gpu);
    options.addOption(bool, "feat_tui", feat_tui);
    options.addOption(bool, "feat_accelerator", feat_accelerator);
    options.addOption(bool, "feat_shader", feat_shader);
    options.addOption(bool, "feat_mlir", feat_mlir);
    options.addOption(bool, "feat_mobile", feat_mobile);
    options.addOption(bool, "feat_wdbx", feat_wdbx);
    options.addOption(bool, "feat_os_control", feat_os_control);
    const options_mod = options.createModule();

    // Plugin Registry Generation
    const gen_plugin_registry = b.addExecutable(.{
        .name = "gen_plugin_registry",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/generate_plugin_registry.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_gen_plugin_registry = b.addRunArtifact(gen_plugin_registry);
    run_gen_plugin_registry.addArg("src/plugins");
    run_gen_plugin_registry.addArg("src/plugin_registry.zig");

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
    exe.step.dependOn(&run_gen_plugin_registry.step);
    b.installArtifact(exe);

    const mcp_exe = b.addExecutable(.{
        .name = "abi-mcp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mcp/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    b.installArtifact(mcp_exe);

    // Steps
    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const cli_step = b.step("cli", "Build ABI CLI");
    cli_step.dependOn(b.getInstallStep());

    const mcp_step = b.step("mcp", "Build MCP server");
    mcp_step.dependOn(&b.addInstallArtifact(mcp_exe, .{}).step);

    // Tests
    const mod_tests = b.addTest(.{
        .root_module = abi_mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const connector_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/connectors/mod.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_connector_tests = b.addRunArtifact(connector_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_connector_tests.step);

    // Integration Tests
    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/integration_tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);

    const test_integration_step = b.step("test-integration", "Run integration tests");
    test_integration_step.dependOn(&run_integration_tests.step);

    // Benchmarks
    const benchmarks = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/benchmarks.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    const run_benchmarks = b.addRunArtifact(benchmarks);

    const bench_step = b.step("benchmarks", "Run benchmark suite");
    bench_step.dependOn(&run_benchmarks.step);

    // Fmt and Parity Checks
    const fmt_check = b.addSystemCommand(&.{ "zig", "fmt", "--check", "src", "tools", "build.zig" });
    const fmt = b.addSystemCommand(&.{ "zig", "fmt", "src", "tools", "build.zig" });

    const parity_exe = b.addExecutable(.{
        .name = "check_parity",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/check_parity.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const parity_check = b.addRunArtifact(parity_exe);

    const check_step = b.step("check", "Run all checks (build + tests + lint + parity)");
    check_step.dependOn(&exe.step);
    check_step.dependOn(&mcp_exe.step);
    check_step.dependOn(test_step);
    check_step.dependOn(&fmt_check.step);
    check_step.dependOn(&parity_check.step);

    const full_check_step = b.step("full-check", "Run the full validation suite");
    full_check_step.dependOn(check_step);

    const lint_step = b.step("lint", "Check Zig formatting");
    lint_step.dependOn(&fmt_check.step);

    const fix_step = b.step("fix", "Format Zig sources");
    fix_step.dependOn(&fmt.step);

    const check_parity_step = b.step("check-parity", "Check feature mod/stub API parity");
    check_parity_step.dependOn(&parity_check.step);
}
