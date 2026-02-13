const std = @import("std");
const builtin = @import("builtin");

// Modular build system
const options_mod = @import("build/options.zig");
const modules = @import("build/modules.zig");
const flags = @import("build/flags.zig");
const targets = @import("build/targets.zig");
const mobile = @import("build/mobile.zig");
const wasm = @import("build/wasm.zig");
const gpu = @import("build/gpu.zig");

// Re-export for external use
pub const GpuBackend = gpu.GpuBackend;
pub const BuildOptions = options_mod.BuildOptions;

comptime {
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @compileError(std.fmt.comptimePrint(
            "ABI requires Zig 0.16.0 or newer (detected {d}.{d}.{d}).\nUse `zvm use master` or install from ziglang.org/builds.",
            .{
                builtin.zig_version.major,
                builtin.zig_version.minor,
                builtin.zig_version.patch,
            },
        ));
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const options = options_mod.readBuildOptions(b);
    options_mod.validateOptions(options);

    const build_opts = modules.createBuildOptionsModule(b, options);
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_opts);

    // ── CLI executable ───────────────────────────────────────────────────
    const cli_path = if (targets.pathExists(b, "tools/cli/main.zig"))
        "tools/cli/main.zig"
    else
        "src/api/main.zig";
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path(cli_path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exe.root_module.addImport("abi", abi_module);
    if (targets.pathExists(b, "tools/cli/main.zig"))
        exe.root_module.addImport("cli", modules.createCliModule(b, abi_module, target, optimize));
    targets.applyPerformanceTweaks(exe, optimize);
    b.installArtifact(exe);

    const run_cli = b.addRunArtifact(exe);
    if (b.args) |args| run_cli.addArgs(args);
    b.step("run", "Run the ABI CLI").dependOn(&run_cli.step);

    // ── Examples & benchmarks (table-driven) ─────────────────────────────
    const examples_step = b.step("examples", "Build all examples");
    targets.buildTargets(b, &targets.example_targets, abi_module, build_opts, target, optimize, examples_step, false);

    // ── CLI smoke tests ──────────────────────────────────────────────────
    const cli_tests_step = b.step("cli-tests", "Run smoke test of CLI commands");
    const cli_commands = [_][]const []const u8{
        &.{"--help"},
        &.{"--version"},
        &.{"system-info"},
        &.{ "db", "stats" },
        &.{ "gpu", "status" },
        &.{ "task", "list" },
        &.{ "config", "show" },
    };
    for (cli_commands) |args| {
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.addArgs(args);
        cli_tests_step.dependOn(&run_cmd.step);
    }

    // ── Lint ─────────────────────────────────────────────────────────────
    const lint_fmt = b.addFmt(.{ .paths = &.{"."}, .check = true });
    b.step("lint", "Check code formatting").dependOn(&lint_fmt.step);

    // ── Tests ────────────────────────────────────────────────────────────
    var test_step: ?*std.Build.Step = null;
    if (targets.pathExists(b, "src/services/tests/mod.zig")) {
        const tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/services/tests/mod.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        tests.root_module.addImport("abi", abi_module);
        tests.root_module.addImport("build_options", build_opts);
        b.step("typecheck", "Compile tests without running").dependOn(&tests.step);
        const run_tests = b.addRunArtifact(tests);
        run_tests.skip_foreign_checks = true;
        test_step = b.step("test", "Run unit tests");
        test_step.?.dependOn(&run_tests.step);
    }

    // ── Feature tests ────────────────────────────────────────────────────
    var feature_tests_step: ?*std.Build.Step = null;
    if (targets.pathExists(b, "src/feature_test_root.zig")) {
        const feature_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/feature_test_root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        feature_tests.root_module.addImport("build_options", build_opts);
        const run_feature_tests = b.addRunArtifact(feature_tests);
        run_feature_tests.skip_foreign_checks = true;
        const ft_step = b.step("feature-tests", "Run feature module inline tests");
        ft_step.dependOn(&run_feature_tests.step);
        feature_tests_step = &run_feature_tests.step;
    }

    // ── Flag validation matrix ───────────────────────────────────────────
    const validate_flags_step = flags.addFlagValidation(b, target, optimize);

    // ── Import rule check ────────────────────────────────────────────────
    const import_check = b.addSystemCommand(&.{ "bash", "scripts/check_import_rules.sh" });
    const import_check_step = b.step("check-imports", "Verify no @import(\"abi\") in feature modules");
    import_check_step.dependOn(&import_check.step);

    // ── Full check ───────────────────────────────────────────────────────
    const full_check_step = b.step("full-check", "Run formatting, unit tests, CLI smoke tests, and flag validation");
    full_check_step.dependOn(&lint_fmt.step);
    if (test_step) |ts| full_check_step.dependOn(ts);
    full_check_step.dependOn(cli_tests_step);
    full_check_step.dependOn(validate_flags_step);
    full_check_step.dependOn(&import_check.step);
    if (feature_tests_step) |fts| full_check_step.dependOn(fts);

    // ── Benchmarks ───────────────────────────────────────────────────────
    const bench_all_step = b.step("bench-all", "Run all benchmark suites");
    targets.buildTargets(b, &targets.benchmark_targets, abi_module, build_opts, target, optimize, bench_all_step, true);

    // ── Documentation ────────────────────────────────────────────────────
    if (targets.pathExists(b, "tools/gendocs/main.zig")) {
        const gendocs = b.addExecutable(.{
            .name = "gendocs",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/gendocs/main.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        const run_gendocs = b.addRunArtifact(gendocs);
        if (b.args) |args| run_gendocs.addArgs(args);
        b.step("gendocs", "Generate API documentation").dependOn(&run_gendocs.step);
    }

    if (targets.pathExists(b, "tools/docs_site/main.zig")) {
        const docs_site = b.addExecutable(.{
            .name = "docs-site",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/docs_site/main.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        const run_docs_site = b.addRunArtifact(docs_site);
        if (b.args) |args| run_docs_site.addArgs(args);
        b.step("docs-site", "Generate documentation website").dependOn(&run_docs_site.step);
    }

    // ── Profile build ────────────────────────────────────────────────────
    if (targets.pathExists(b, "tools/cli/main.zig")) {
        var profile_opts = options;
        profile_opts.enable_profiling = true;
        const abi_profile = modules.createAbiModule(b, profile_opts, target, optimize);
        const profile_exe = b.addExecutable(.{
            .name = "abi-profile",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/cli/main.zig"),
                .target = target,
                .optimize = .ReleaseFast,
                .link_libc = true,
            }),
        });
        profile_exe.root_module.addImport("abi", abi_profile);
        profile_exe.root_module.addImport("cli", modules.createCliModule(b, abi_profile, target, optimize));
        profile_exe.root_module.strip = false;
        profile_exe.root_module.omit_frame_pointer = false;
        b.installArtifact(profile_exe);
        b.step("profile", "Build with performance profiling").dependOn(b.getInstallStep());
    }

    // ── Mobile ───────────────────────────────────────────────────────────
    _ = mobile.addMobileBuild(b, options, optimize);

    // ── C Library ────────────────────────────────────────────────────────
    if (targets.pathExists(b, "bindings/c/src/abi_c.zig")) {
        const lib = b.addLibrary(.{
            .name = "abi",
            .root_module = b.createModule(.{
                .root_source_file = b.path("bindings/c/src/abi_c.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .linkage = .dynamic,
        });
        lib.root_module.addImport("abi", abi_module);
        lib.root_module.addImport("build_options", build_opts);
        b.step("lib", "Build C shared library").dependOn(&b.addInstallArtifact(lib, .{}).step);

        const header_install = b.addInstallFile(b.path("bindings/c/include/abi.h"), "include/abi.h");
        b.step("c-header", "Install C header file").dependOn(&header_install.step);
    }

    // ── Performance verification tool ────────────────────────────────────
    if (targets.pathExists(b, "tools/perf/check.zig")) {
        const check_perf_exe = b.addExecutable(.{
            .name = "abi-check-perf",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/perf/check.zig"),
                .target = target,
                .optimize = .ReleaseSafe,
            }),
        });
        b.step("check-perf", "Build performance verification tool (pipe benchmark JSON to run)")
            .dependOn(&b.addInstallArtifact(check_perf_exe, .{}).step);
    }

    // ── WASM ─────────────────────────────────────────────────────────────
    const check_wasm_step = wasm.addWasmBuild(b, options, abi_module, optimize);

    // ── Verify-all ───────────────────────────────────────────────────────
    const version_script_run = b.addSystemCommand(&.{ "sh", "scripts/check_zig_version_consistency.sh" });
    const verify_all_step = b.step("verify-all", "full-check + version script + examples + bench-all + check-wasm");
    verify_all_step.dependOn(full_check_step);
    verify_all_step.dependOn(&version_script_run.step);
    verify_all_step.dependOn(examples_step);
    verify_all_step.dependOn(bench_all_step);
    if (check_wasm_step) |s| verify_all_step.dependOn(s);
}
