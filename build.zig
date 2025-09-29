const std = @import("std");

fn createBuildOptions(b: *std.Build, optimize: std.builtin.OptimizeMode) *std.Build.Step.Options {
    const package_version = b.option([]const u8, "package-version", "Override package version used by build options.") orelse "0.1.0";
    const enable_gpu = b.option(bool, "enable-gpu", "Compile with GPU/WebGPU backends enabled.") orelse true;
    const enable_wdbx = b.option(bool, "enable-wdbx", "Compile with the WDBX vector database enabled.") orelse true;
    const enable_http = b.option(bool, "enable-http", "Compile HTTP tooling and server surfaces.") orelse true;
    const embedded_assets = b.option(bool, "embed-assets", "Embed static documentation assets into binaries.") orelse false;

    const opts = b.addOptions();
    opts.addOption([]const u8, "package_version", package_version);
    opts.addOption(bool, "enable_gpu", enable_gpu);
    opts.addOption(bool, "enable_wdbx", enable_wdbx);
    opts.addOption(bool, "enable_http", enable_http);
    opts.addOption(bool, "embed_assets", embedded_assets);
    opts.addOption(std.builtin.OptimizeMode, "optimize_mode", optimize);
    return opts;
}

fn addRunStep(
    b: *std.Build,
    name: []const u8,
    description: []const u8,
    exe: *std.Build.Step.Compile,
    propagate_args: bool,
) void {
    const step = b.step(name, description);
    const run_cmd = b.addRunArtifact(exe);
    if (propagate_args) {
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }
    }
    step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
}

fn addStandaloneTest(
    b: *std.Build,
    test_step: *std.Build.Step,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    abi_mod: *std.Build.Module,
    path: []const u8,
) void {
    const test_compile = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path(path),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    const run_cmd = b.addRunArtifact(test_compile);
    test_step.dependOn(&run_cmd.step);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_opts = createBuildOptions(b, optimize);

    const abi_mod = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_mod.addOptions("build_options", build_opts);

    const main_exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    b.installArtifact(main_exe);
    addRunStep(b, "run", "Run the ABI CLI", main_exe, true);

    const tools_exe = b.addExecutable(.{
        .name = "abi-tools",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    b.installArtifact(tools_exe);
    addRunStep(b, "tools", "Run the consolidated tooling router", tools_exe, true);

    const interactive_cli_exe = b.addExecutable(.{
        .name = "abi-interactive",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/interactive_cli.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    b.installArtifact(interactive_cli_exe);
    addRunStep(b, "cli", "Launch the interactive CLI shell", interactive_cli_exe, true);

    const metrics_exe = b.addExecutable(.{
        .name = "abi-metrics",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/metrics.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    b.installArtifact(metrics_exe);
    addRunStep(b, "metrics", "Export metrics snapshot", metrics_exe, false);

    const agent_demo_exe = b.addExecutable(.{
        .name = "abi-agent-demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/examples/agent_subsystem_demo.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    b.installArtifact(agent_demo_exe);
    addRunStep(b, "agent-demo", "Run the agent subsystem showcase", agent_demo_exe, false);

    const docs_exe = b.addExecutable(.{
        .name = "abi-docs",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/docs_generator.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    b.installArtifact(docs_exe);
    addRunStep(b, "docs", "Generate API and reference documentation", docs_exe, false);

    const bench_exe = b.addExecutable(.{
        .name = "abi-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    b.installArtifact(bench_exe);
    addRunStep(b, "bench", "Run benchmark harness", bench_exe, true);

    const fmt_step = b.addFmt(.{
        .paths = &.{
            "build.zig",
            "build.zig.zon",
            "src",
            "benchmarks",
            "docs",
            "tests",
        },
    });
    b.step("fmt", "Format Zig sources and documentation metadata").dependOn(&fmt_step.step);

    const test_step = b.step("test", "Run unit, CLI, and subsystem tests");

    const core_tests = b.addTest(.{ .root_module = abi_mod });
    test_step.dependOn(&b.addRunArtifact(core_tests).step);

    addStandaloneTest(b, test_step, target, optimize, abi_mod, "src/tools/cli/modern_cli.zig");
    addStandaloneTest(b, test_step, target, optimize, abi_mod, "src/tests/unit/test_ai.zig");
    addStandaloneTest(b, test_step, target, optimize, abi_mod, "src/tests/unit/test_database_hnsw.zig");
    addStandaloneTest(b, test_step, target, optimize, abi_mod, "src/tests/unit/test_gpu.zig");
    addStandaloneTest(b, test_step, target, optimize, abi_mod, "src/tests/unit/test_logging.zig");
    addStandaloneTest(b, test_step, target, optimize, abi_mod, "src/tests/unit/test_utils.zig");
    addStandaloneTest(b, test_step, target, optimize, abi_mod, "src/tests/unit/test_plugin_connector_integration.zig");

    const perf_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/performance_ci.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
            },
        }),
    });
    test_step.dependOn(&b.addRunArtifact(perf_tests).step);
}
