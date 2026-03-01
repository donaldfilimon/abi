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

    const wdbx_module = b.addModule("wdbx", .{
        .root_source_file = b.path("src/features/database/wdbx/wdbx.zig"),
        .target = target,
        .optimize = optimize,
    });

    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/abi.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_opts);
    abi_module.addImport("wdbx", wdbx_module);

    // ── CLI executable ──────────────────────────────────────────────────
    const exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exe.root_module.addImport("abi", abi_module);
    exe.root_module.addImport("cli", modules.createCliModule(b, abi_module, target, optimize));
    targets.applyPerformanceTweaks(exe, optimize);
    link.applyAllPlatformLinks(exe.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends);
    b.installArtifact(exe);

    const run_cli = b.addRunArtifact(exe);
    if (b.args) |args| run_cli.addArgs(args);
    b.step("run", "Run the ABI CLI").dependOn(&run_cli.step);

    const run_editor = b.addRunArtifact(exe);
    run_editor.addArg("ui");
    run_editor.addArg("editor");
    if (b.args) |args| run_editor.addArgs(args);
    b.step("editor", "Run the inline CLI TUI editor").dependOn(&run_editor.step);

    // ── Examples (table-driven) ─────────────────────────────────────────
    const examples_step = b.step("examples", "Build all examples");
    targets.buildTargets(b, &targets.example_targets, abi_module, build_opts, target, optimize, examples_step, false);

    // ── CLI smoke tests ─────────────────────────────────────────────────
    const cli_tests_step = cli_tests.addCliTests(b, exe);

    // ── TUI / CLI unit tests ───────────────────────────────────────────
    var tui_tests_step: ?*std.Build.Step = null;
    const cli_test_mod = b.createModule(.{
        .root_source_file = b.path("tools/cli/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cli_test_mod.addImport("abi", abi_module);
    const cli_tests_artifact = b.addTest(.{ .root_module = cli_test_mod });
    cli_tests_artifact.test_runner = .{
        .path = b.path("build/cli_tui_test_runner.zig"),
        .mode = .simple,
    };
    link.applyAllPlatformLinks(cli_tests_artifact.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends);
    const run_cli_tests = b.addRunArtifact(cli_tests_artifact);
    run_cli_tests.skip_foreign_checks = true;
    tui_tests_step = b.step("tui-tests", "Run TUI and CLI unit tests");
    tui_tests_step.?.dependOn(&run_cli_tests.step);

    // ── Lint / format ───────────────────────────────────────────────────
    const fmt_paths = &.{ "build.zig", "build", "src", "tools", "examples" };
    const lint_fmt = b.addFmt(.{ .paths = fmt_paths, .check = true });
    b.step("lint", "Check code formatting").dependOn(&lint_fmt.step);
    const fix_fmt = b.addFmt(.{ .paths = fmt_paths, .check = false });
    b.step("fix", "Format source files in place").dependOn(&fix_fmt.step);

    // ── Tests ───────────────────────────────────────────────────────────
    var test_step: ?*std.Build.Step = null;
    var typecheck_step: ?*std.Build.Step = null;
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
        link.applyAllPlatformLinks(tests.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends);
        typecheck_step = b.step("typecheck", "Compile tests without running");
        typecheck_step.?.dependOn(&tests.step);
        const run_tests = b.addRunArtifact(tests);
        run_tests.skip_foreign_checks = true;
        test_step = b.step("test", "Run unit tests");
        test_step.?.dependOn(&run_tests.step);
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

    const workflow_contract_step = b.step("check-workflow-orchestration", "Advisory workflow-orchestration contract checks");
    workflow_contract_step.dependOn(&addScriptRunner(
        b,
        "abi-check-workflow-orchestration",
        "tools/scripts/check_workflow_orchestration.zig",
        target,
        optimize,
    ).step);

    const workflow_contract_strict_step = b.step("check-workflow-orchestration-strict", "Strict workflow-orchestration contract checks");
    const workflow_contract_strict = addScriptRunner(
        b,
        "abi-check-workflow-orchestration-strict",
        "tools/scripts/check_workflow_orchestration.zig",
        target,
        optimize,
    );
    workflow_contract_strict.addArg("--strict");
    workflow_contract_strict_step.dependOn(&workflow_contract_strict.step);

    // ── CLI DSL registry/codegen ───────────────────────────────────────
    const generate_cli_registry = addScriptRunner(
        b,
        "abi-generate-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
    );
    generate_cli_registry.addArg("--output");
    generate_cli_registry.addArg(".zig-cache/abi/generated/cli_registry.zig");
    const generate_cli_registry_step = b.step("generate-cli-registry", "Generate CLI registry artifact in build cache");
    generate_cli_registry_step.dependOn(&generate_cli_registry.step);

    const refresh_cli_registry = addScriptRunner(
        b,
        "abi-refresh-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
    );
    refresh_cli_registry.addArg("--snapshot");
    const refresh_cli_registry_step = b.step("refresh-cli-registry", "Refresh tracked CLI registry snapshot");
    refresh_cli_registry_step.dependOn(&refresh_cli_registry.step);

    const check_cli_registry = addScriptRunner(
        b,
        "abi-check-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
    );
    check_cli_registry.addArg("--check");
    check_cli_registry.addArg("--snapshot");
    const check_cli_registry_step = b.step("check-cli-registry", "Check CLI registry snapshot determinism");
    check_cli_registry_step.dependOn(&check_cli_registry.step);

    const check_cli_dsl_consistency_step = b.step("check-cli-dsl-consistency", "Verify CLI/TUI DSL organization contracts");
    check_cli_dsl_consistency_step.dependOn(&addScriptRunner(
        b,
        "abi-check-cli-dsl-consistency",
        "tools/scripts/check_cli_dsl_consistency.zig",
        target,
        optimize,
    ).step);

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
    full_check_step.dependOn(check_cli_registry_step);
    full_check_step.dependOn(check_cli_dsl_consistency_step);
    if (tui_tests_step) |step| full_check_step.dependOn(step);

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
        run_check_docs.addArg("--untracked-md");
        const docs_check = b.step("check-docs", "Validate docs generator determinism and output policy");
        docs_check.dependOn(&run_check_docs.step);
        check_docs_step = docs_check;
    }

    // ── Profile build ───────────────────────────────────────────────────
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
    link.applyAllPlatformLinks(profile_mod, target.result.os.tag, options.gpu_metal(), options.gpu_backends);
    b.installArtifact(profile_exe);
    b.step("profile", "Build with performance profiling").dependOn(b.getInstallStep());

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

    // ── Cross-compilation platform matrix ────────────────────────────────
    const cross_check_step = b.step(
        "cross-check",
        "Verify the abi module compiles for all supported platform targets",
    );
    const cross_targets = targets.cross_check_targets;
    inline for (cross_targets) |ct| {
        const cross_target = b.resolveTargetQuery(.{
            .cpu_arch = ct.arch,
            .os_tag = ct.os,
            .abi = ct.abi,
        });
        var cross_opts = options;
        // Disable features that cannot compile for non-native targets
        cross_opts.enable_mobile = false;
        if (ct.os == .wasi or ct.os == .freestanding or ct.os == .emscripten) {
            cross_opts.enable_database = false;
            cross_opts.enable_network = false;
            cross_opts.enable_gpu = false;
            cross_opts.enable_profiling = false;
            cross_opts.enable_web = false;
            cross_opts.enable_cloud = false;
            cross_opts.enable_storage = false;
            cross_opts.gpu_backends = &.{};
        } else {
            cross_opts.gpu_backends = &.{.stdgpu};
        }
        const cross_build_opts = modules.createBuildOptionsModule(b, cross_opts);
        const cross_abi_mod = b.createModule(.{
            .root_source_file = b.path("src/abi.zig"),
            .target = cross_target,
            .optimize = optimize,
        });
        cross_abi_mod.addImport("build_options", cross_build_opts);
        const cross_lib = b.addLibrary(.{
            .name = "cross-" ++ ct.name,
            .root_module = cross_abi_mod,
            .linkage = .static,
        });
        cross_check_step.dependOn(&cross_lib.step);
    }

    // ── Verify-all ──────────────────────────────────────────────────────
    const gate_hardening_step = b.step("gate-hardening", "Run deterministic gate hardening checks");
    gate_hardening_step.dependOn(toolchain_doctor_step);
    if (typecheck_step) |step| gate_hardening_step.dependOn(step);
    gate_hardening_step.dependOn(cli_tests_step);
    if (tui_tests_step) |step| gate_hardening_step.dependOn(step);
    gate_hardening_step.dependOn(check_cli_registry_step);
    if (check_docs_step) |step| gate_hardening_step.dependOn(step);
    gate_hardening_step.dependOn(workflow_contract_strict_step);
    gate_hardening_step.dependOn(full_check_step);

    const verify_all_step = b.step("verify-all", "full-check + consistency + feature-tests + examples + check-wasm + cross-check");
    verify_all_step.dependOn(full_check_step);
    verify_all_step.dependOn(consistency_step);
    if (feature_tests_step) |fts| verify_all_step.dependOn(fts);
    verify_all_step.dependOn(examples_step);
    if (check_wasm_step) |s| verify_all_step.dependOn(s);
    if (check_docs_step) |docs_step| verify_all_step.dependOn(docs_step);
    verify_all_step.dependOn(cross_check_step);
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
