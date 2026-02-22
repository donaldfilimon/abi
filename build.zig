const std = @import("std");
const builtin = @import("builtin");

// ── Modular build system ────────────────────────────────────────────────
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
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16)
        @compileError(std.fmt.comptimePrint(
            "ABI requires Zig 0.16.0 or newer (detected {d}.{d}.{d}). " ++
                "Use `zvm use master` then align to `.zigversion`.",
            .{ builtin.zig_version.major, builtin.zig_version.minor, builtin.zig_version.patch },
        ));
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const can_link_metal = link.canLinkMetalFrameworks(b.graph.io, target.result.os.tag);

    // ── Build options ───────────────────────────────────────────────────
    const backend_arg = b.option([]const u8, "gpu-backend", gpu.backend_option_help);
    const cli_full_env_file = b.option(
        []const u8,
        "cli-full-env-file",
        "Path to KEY=VALUE env file for exhaustive CLI integration tests",
    );
    const cli_full_allow_blocked = b.option(
        bool,
        "cli-full-allow-blocked",
        "Continue cli-tests-full when preflight checks fail and mark blocked rows",
    ) orelse false;
    const cli_full_timeout_scale = b.option(
        f64,
        "cli-full-timeout-scale",
        "Timeout multiplier for cli-tests-full on slower hosts",
    ) orelse 1.0;

    link.validateMetalBackendRequest(b, backend_arg, target.result.os.tag, can_link_metal);
    const options = options_mod.readBuildOptions(
        b,
        target.result.os.tag,
        target.result.abi,
        can_link_metal,
        backend_arg,
    );
    options_mod.validateOptions(options);

    // ── Core modules ────────────────────────────────────────────────────
    const build_opts = modules.createBuildOptionsModule(b, options);
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_opts);

    // ── CLI executable ──────────────────────────────────────────────────
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

    // ── Examples (table-driven) ─────────────────────────────────────────
    const examples_step = b.step("examples", "Build all examples");
    targets.buildTargets(b, &targets.example_targets, abi_module, build_opts, target, optimize, examples_step, false);

    // ── CLI smoke tests ─────────────────────────────────────────────────
    const cli_tests_step = cli_tests.addCliTests(b, exe);
    _ = cli_tests.addCliTestsFull(b, .{
        .env_file = cli_full_env_file,
        .allow_blocked = cli_full_allow_blocked,
        .timeout_scale = cli_full_timeout_scale,
    });

    // ── Lint / format ───────────────────────────────────────────────────
    const fmt_paths = &.{ "build.zig", "build", "src", "tools", "examples" };
    const lint_fmt = b.addFmt(.{ .paths = fmt_paths, .check = true });
    b.step("lint", "Check code formatting").dependOn(&lint_fmt.step);
    const fix_fmt = b.addFmt(.{ .paths = fmt_paths, .check = false });
    b.step("fix", "Format source files in place").dependOn(&fix_fmt.step);

    // ── Tests ───────────────────────────────────────────────────────────
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

    // ── vNext compatibility tests ───────────────────────────────────────
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

    // ── Feature tests (manifest-driven) ─────────────────────────────────
    const feature_tests_step = test_discovery.addFeatureTests(b, options, build_opts, target, optimize);

    // ── Flag validation matrix ──────────────────────────────────────────
    const validate_flags_step = flags.addFlagValidation(b, target, optimize);

    // ── Import rule check ───────────────────────────────────────────────
    const import_check_step = b.step("check-imports", "Verify no @import(\"abi\") in feature modules");
    import_check_step.dependOn(&addScriptRunner(b, "abi-check-import-rules", "tools/scripts/check_import_rules.zig", target, optimize).step);

    // ── Consistency checks ──────────────────────────────────────────────
    const toolchain_doctor_step = b.step("toolchain-doctor", "Diagnose local Zig PATH/version drift against repository pin");
    toolchain_doctor_step.dependOn(&addScriptRunner(b, "abi-toolchain-doctor", "tools/scripts/toolchain_doctor.zig", target, optimize).step);

    const consistency_step = b.step("check-consistency", "Verify Zig version/baseline consistency and Zig 0.16 conformance patterns");
    consistency_step.dependOn(&addScriptRunner(b, "abi-check-zig-version-consistency", "tools/scripts/check_zig_version_consistency.zig", target, optimize).step);
    consistency_step.dependOn(&addScriptRunner(b, "abi-check-test-baseline-consistency", "tools/scripts/check_test_baseline_consistency.zig", target, optimize).step);
    consistency_step.dependOn(&addScriptRunner(b, "abi-check-zig-016-patterns", "tools/scripts/check_zig_016_patterns.zig", target, optimize).step);
    consistency_step.dependOn(&addScriptRunner(b, "abi-check-feature-catalog", "tools/scripts/check_feature_catalog.zig", target, optimize).step);

    if (targets.pathExists(b, "tools/scripts/check_gpu_policy_consistency.zig")) {
        const gpu_policy_check_module = b.createModule(.{
            .root_source_file = b.path("tools/scripts/check_gpu_policy_consistency.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        gpu_policy_check_module.addImport("build_gpu_policy", b.createModule(.{
            .root_source_file = b.path("build/gpu_policy.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }));
        gpu_policy_check_module.addImport("runtime_gpu_policy", b.createModule(.{
            .root_source_file = b.path("src/features/gpu/policy/mod.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }));
        const check_gpu_policy_exe = b.addExecutable(.{
            .name = "abi-check-gpu-policy-consistency",
            .root_module = gpu_policy_check_module,
        });
        consistency_step.dependOn(&b.addRunArtifact(check_gpu_policy_exe).step);
    }

    const ralph_gate_step = b.step("ralph-gate", "Require live Ralph scoring report and threshold pass");
    ralph_gate_step.dependOn(&addScriptRunner(b, "abi-check-ralph-gate", "tools/scripts/check_ralph_gate.zig", target, optimize).step);

    // ── Baseline validation ─────────────────────────────────────────────
    const validate_baseline_step = b.step("validate-baseline", "Run tests and verify counts match tools/scripts/baseline.zig");
    const validate_baseline = addScriptRunner(b, "abi-validate-test-counts", "tools/scripts/validate_test_counts.zig", target, optimize);
    validate_baseline.addArg("--main-only");
    if (test_step) |ts| validate_baseline.step.dependOn(ts);
    validate_baseline_step.dependOn(&validate_baseline.step);

    // ── Full check ──────────────────────────────────────────────────────
    const full_check_step = b.step("full-check", "Run formatting, unit tests, CLI smoke tests, and flag validation");
    full_check_step.dependOn(&lint_fmt.step);
    if (test_step) |ts| full_check_step.dependOn(ts);
    full_check_step.dependOn(cli_tests_step);
    full_check_step.dependOn(validate_flags_step);
    full_check_step.dependOn(import_check_step);
    full_check_step.dependOn(consistency_step);
    if (vnext_compat_step) |step| full_check_step.dependOn(step);

    var check_docs_step: ?*std.Build.Step = null;

    // ── Documentation ───────────────────────────────────────────────────
    if (targets.pathExists(b, "tools/gendocs/main.zig")) {
        const gendocs_module = b.createModule(.{
            .root_source_file = b.path("tools/gendocs/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        gendocs_module.addImport("roadmap_catalog", b.createModule(.{
            .root_source_file = b.path("src/services/tasks/roadmap_catalog.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }));

        const gendocs = b.addExecutable(.{
            .name = "gendocs",
            .root_module = gendocs_module,
        });
        const run_gendocs = b.addRunArtifact(gendocs);
        if (b.args) |args| run_gendocs.addArgs(args);
        b.step("gendocs", "Generate docs/api, docs/_docs, docs/plans, and docs/api-app").dependOn(&run_gendocs.step);

        const run_check_docs = b.addRunArtifact(gendocs);
        run_check_docs.addArg("--check");
        const docs_check = b.step("check-docs", "Validate docs outputs are up to date");
        docs_check.dependOn(&run_check_docs.step);
        check_docs_step = docs_check;
    }

    // ── Profile build ───────────────────────────────────────────────────
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

    // ── Mobile ──────────────────────────────────────────────────────────
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

    // ── Performance verification tool ───────────────────────────────────
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

    // ── WASM ────────────────────────────────────────────────────────────
    const check_wasm_step = wasm.addWasmBuild(b, options, abi_module, optimize);

    // ── Verify-all ──────────────────────────────────────────────────────
    const verify_all_step = b.step("verify-all", "full-check + consistency + feature-tests + examples + check-wasm");
    verify_all_step.dependOn(full_check_step);
    verify_all_step.dependOn(consistency_step);
    if (feature_tests_step) |fts| verify_all_step.dependOn(fts);
    verify_all_step.dependOn(examples_step);
    if (check_wasm_step) |s| verify_all_step.dependOn(s);
    if (check_docs_step) |docs_step| verify_all_step.dependOn(docs_step);
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Build and run a standalone Zig script (used for consistency checks, gate
/// checks, etc.).  Returns the `Run` step so callers can add arguments or
/// set dependencies.
fn addScriptRunner(
    b: *std.Build,
    name: []const u8,
    source: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Run {
    const exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(source),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    return b.addRunArtifact(exe);
}
