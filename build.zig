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
const link = @import("build/link.zig");
const cli_tests = @import("build/cli_tests.zig");
const test_discovery = @import("build/test_discovery.zig");

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
    link.applyFrameworkLinks(exe.root_module, target.result.os.tag, options.gpu_metal());
    b.installArtifact(exe);

    const run_cli = b.addRunArtifact(exe);
    if (b.args) |args| run_cli.addArgs(args);
    b.step("run", "Run the ABI CLI").dependOn(&run_cli.step);

    // ── Examples (table-driven) ────────────────────────────────────────
    const examples_step = b.step("examples", "Build all examples");
    targets.buildTargets(b, &targets.example_targets, abi_module, build_opts, target, optimize, examples_step, false);

    // ── CLI smoke tests ──────────────────────────────────────────────────
    const cli_tests_step = cli_tests.addCliTests(b, exe);

    // ── Lint ─────────────────────────────────────────────────────────────
    const fmt_paths = &.{
        "build.zig",
        "build",
        "src",
        "tools",
        "examples",
    };
    const lint_fmt = b.addFmt(.{
        .paths = fmt_paths,
        .check = true,
    });
    b.step("lint", "Check code formatting").dependOn(&lint_fmt.step);

    const fix_fmt = b.addFmt(.{
        .paths = fmt_paths,
        .check = false,
    });
    b.step("fix", "Format source files in place").dependOn(&fix_fmt.step);

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
        link.applyFrameworkLinks(tests.root_module, target.result.os.tag, options.gpu_metal());
        b.step("typecheck", "Compile tests without running").dependOn(&tests.step);
        const run_tests = b.addRunArtifact(tests);
        run_tests.skip_foreign_checks = true;
        test_step = b.step("test", "Run unit tests");
        test_step.?.dependOn(&run_tests.step);
    }

    // ── vNext compatibility tests ──────────────────────────────────────
    var vnext_compat_step: ?*std.Build.Step = null;
    if (targets.pathExists(b, "src/services/tests/vnext_compat_test.zig")) {
        const vnext_compat = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/services/tests/vnext_compat_test.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        vnext_compat.root_module.addImport("abi", abi_module);
        vnext_compat.root_module.addImport("build_options", build_opts);
        const run_vnext_compat = b.addRunArtifact(vnext_compat);
        run_vnext_compat.skip_foreign_checks = true;
        vnext_compat_step = b.step("vnext-compat", "Run vnext compatibility tests");
        vnext_compat_step.?.dependOn(&run_vnext_compat.step);
    }

    // ── Feature tests (manifest-driven; see build/test_discovery.zig) ───
    const feature_tests_step = test_discovery.addFeatureTests(b, options, build_opts, target, optimize);

    // ── Flag validation matrix ───────────────────────────────────────────
    const validate_flags_step = flags.addFlagValidation(b, target, optimize);

    // ── Import rule check ────────────────────────────────────────────────
    const import_check = b.addSystemCommand(&.{ "bash", "scripts/check_import_rules.sh" });
    const import_check_step = b.step("check-imports", "Verify no @import(\"abi\") in feature modules");
    import_check_step.dependOn(&import_check.step);

    // ── Consistency checks ───────────────────────────────────────────────
    const toolchain_doctor = b.addSystemCommand(&.{ "bash", "scripts/toolchain_doctor.sh" });
    const toolchain_doctor_step = b.step(
        "toolchain-doctor",
        "Diagnose local Zig PATH/version drift against repository pin",
    );
    toolchain_doctor_step.dependOn(&toolchain_doctor.step);

    const check_versions = b.addSystemCommand(&.{ "bash", "scripts/check_zig_version_consistency.sh" });
    const check_baselines = b.addSystemCommand(&.{ "bash", "scripts/check_test_baseline_consistency.sh" });
    const check_patterns = b.addSystemCommand(&.{ "bash", "scripts/check_zig_016_patterns.sh" });
    const check_features = b.addSystemCommand(&.{ "bash", "scripts/check_feature_catalog.sh" });
    const check_ralph = b.addSystemCommand(&.{ "bash", "scripts/check_ralph_gate.sh" });
    const consistency_step = b.step(
        "check-consistency",
        "Verify Zig version/baseline consistency and Zig 0.16 conformance patterns",
    );
    consistency_step.dependOn(&check_versions.step);
    consistency_step.dependOn(&check_baselines.step);
    consistency_step.dependOn(&check_patterns.step);
    consistency_step.dependOn(&check_features.step);

    const ralph_gate_step = b.step("ralph-gate", "Require live Ralph scoring report and threshold pass");
    ralph_gate_step.dependOn(&check_ralph.step);

    // ── Baseline validation ─────────────────────────────────────────────
    const validate_baseline = b.addSystemCommand(&.{ "bash", "scripts/validate_test_counts.sh", "--main-only" });
    if (test_step) |ts| validate_baseline.step.dependOn(ts);
    const validate_baseline_step = b.step(
        "validate-baseline",
        "Run tests and verify counts match scripts/project_baseline.env",
    );
    validate_baseline_step.dependOn(&validate_baseline.step);

    // ── Full check ───────────────────────────────────────────────────────
    const full_check_step = b.step("full-check", "Run formatting, unit tests, CLI smoke tests, and flag validation");
    full_check_step.dependOn(&lint_fmt.step);
    if (test_step) |ts| full_check_step.dependOn(ts);
    full_check_step.dependOn(cli_tests_step);
    full_check_step.dependOn(validate_flags_step);
    full_check_step.dependOn(&import_check.step);
    full_check_step.dependOn(consistency_step);
    if (vnext_compat_step) |step| full_check_step.dependOn(step);
    if (feature_tests_step) |fts| full_check_step.dependOn(fts);

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
        const profile_mod = profile_exe.root_module;
        profile_mod.addImport("abi", abi_profile);
        profile_mod.addImport("cli", modules.createCliModule(b, abi_profile, target, optimize));
        profile_mod.strip = false;
        profile_mod.omit_frame_pointer = false;
        link.applyFrameworkLinks(profile_mod, target.result.os.tag, options.gpu_metal());
        b.installArtifact(profile_exe);
        b.step("profile", "Build with performance profiling").dependOn(b.getInstallStep());
    }

    // ── Mobile ───────────────────────────────────────────────────────────
    _ = mobile.addMobileBuild(b, options, optimize);

    // ── C Library ────────────────────────────────────────────────────────
    const c_bindings_src = "bindings/c/src/abi_c.zig";
    const c_bindings_header = "bindings/c/include/abi.h";
    if (targets.pathExists(b, c_bindings_src) and targets.pathExists(b, c_bindings_header)) {
        const lib = b.addLibrary(.{
            .name = "abi",
            .root_module = b.createModule(.{
                .root_source_file = b.path(c_bindings_src),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .linkage = .dynamic,
        });
        lib.root_module.addImport("abi", abi_module);
        lib.root_module.addImport("build_options", build_opts);
        b.step("lib", "Build C shared library").dependOn(&b.addInstallArtifact(lib, .{}).step);

        const header_install = b.addInstallFile(b.path(c_bindings_header), "include/abi.h");
        b.step("c-header", "Install C header file").dependOn(&header_install.step);
    } else {
        // Keep CI/build graph targets available even when optional C bindings are absent.
        _ = b.step("lib", "Build C shared library (C bindings unavailable in this checkout)");

        const generated = b.addWriteFiles();
        const placeholder_header = generated.add("abi.h",
            \\/* ABI C header placeholder.
            \\ * Full C bindings are not present in this checkout.
            \\ */
            \\#pragma once
            \\
        );
        const header_install = b.addInstallFile(placeholder_header, "include/abi.h");
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
    const verify_all_step = b.step("verify-all", "full-check + consistency checks + examples + check-wasm");
    verify_all_step.dependOn(full_check_step);
    verify_all_step.dependOn(consistency_step);
    verify_all_step.dependOn(examples_step);
    if (check_wasm_step) |s| verify_all_step.dependOn(s);
}
