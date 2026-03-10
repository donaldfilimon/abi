//! ABI build root. Zig 0.16 Build API: root_module, createModule, b.path (LazyPath).
const std = @import("std");
const builtin = @import("builtin");

const cel = @import("build/cel.zig");
const cli_tests = @import("build/cli_tests.zig");
const flags = @import("build/flags.zig");
const gpu = @import("build/gpu.zig");
pub const GpuBackend = gpu.GpuBackend;
const link = @import("build/link.zig");
const mobile = @import("build/mobile.zig");
const modules = @import("build/modules.zig");
const options_mod = @import("build/options.zig");
pub const BuildOptions = options_mod.BuildOptions;
const targets = @import("build/targets.zig");
const test_discovery = @import("build/test_discovery.zig");
const wasm = @import("build/wasm.zig");

// ── Modular build system ────────────────────────────────────────────────
// Re-export for external use
const is_blocked_darwin = builtin.os.tag == .macos and builtin.os.version_range.semver.min.major >= 26;

pub fn build(b: *std.Build) void {
    const target = resolveNativeTarget(b);
    const optimize = b.standardOptimizeOption(.{});
    const can_link_metal = link.canLinkMetalFrameworks(b.graph.io, target.result.os.tag);

    // ── Build options ───────────────────────────────────────────────────
    const backend_arg = b.option([]const u8, "gpu-backend", gpu.backend_option_help);

    link.validateMetalBackendRequest(b, backend_arg, target.result.os.tag, can_link_metal);
    var options = options_mod.readBuildOptions(
        b,
        target.result.os.tag,
        target.result.abi,
        can_link_metal,
        backend_arg,
    );
    if (is_blocked_darwin) {
        options.feat_database = false;
        options.feat_ai = false;
        options.feat_explore = false;
        options.feat_llm = false;
        options.feat_vision = false;
        options.feat_training = false;
        options.feat_reasoning = false;

        // Emit platform-appropriate CEL suggestion
        const cel_status = cel.detectCelStatus(b);
        cel.emitCelSuggestion(b, cel_status);
    }
    options_mod.validateOptions(options);

    // ── Core modules ────────────────────────────────────────────────────
    const build_opts = modules.createBuildOptionsModule(b, options);

    const shared_services_module = modules.createSharedServicesModule(b, build_opts, target, optimize);

    const util_module = b.addModule("util", .{
        .root_source_file = b.path("tools/scripts/util.zig"),
        .target = target,
        .optimize = optimize,
    });

    const core_module = modules.createCoreModule(b, target, optimize, build_opts);

    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    modules.wireAbiImports(abi_module, build_opts, shared_services_module, core_module);

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
    if (is_blocked_darwin) {
        exe.use_llvm = true;
        // Compile-only: don't install or run (linker is broken)
        b.step("run", "Run the ABI CLI (unavailable on blocked Darwin)").dependOn(&exe.step);
        b.step("editor", "Run the inline CLI TUI editor (unavailable on blocked Darwin)").dependOn(&exe.step);
    } else {
        b.installArtifact(exe);

        const run_cli = b.addRunArtifact(exe);
        if (b.args) |args| run_cli.addArgs(args);
        b.step("run", "Run the ABI CLI").dependOn(&run_cli.step);

        const run_editor = b.addRunArtifact(exe);
        run_editor.addArg("ui");
        run_editor.addArg("editor");
        if (b.args) |args| run_editor.addArgs(args);
        b.step("editor", "Run the inline CLI TUI editor").dependOn(&run_editor.step);
    }

    // ── Examples (table-driven) ─────────────────────────────────────────
    const examples_step = b.step("examples", "Build all examples");
    targets.buildTargets(b, &targets.example_targets, abi_module, build_opts, target, optimize, examples_step, false);

    // ── CLI smoke tests ─────────────────────────────────────────────────
    const cli_tests_step = cli_tests.addCliTests(b, exe, abi_module, target, optimize);

    // ── TUI / CLI unit tests ───────────────────────────────────────────
    var tui_tests_step: ?*std.Build.Step = null;
    var gendocs_source_tests_step: ?*std.Build.Step = null;
    var launcher_tests_step: ?*std.Build.Step = null;
    const cli_root_mod = b.createModule(.{
        .root_source_file = b.path("tools/cli/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cli_root_mod.addImport("abi", abi_module);
    cli_root_mod.addImport("shared_services", shared_services_module);

    const cli_test_mod = b.createModule(.{
        .root_source_file = b.path("build/cli_tui_tests_root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cli_test_mod.addImport("cli_root", cli_root_mod);

    const cli_tests_artifact = b.addTest(.{ .root_module = cli_test_mod });
    cli_tests_artifact.test_runner = .{
        .path = b.path("build/cli_tui_test_runner.zig"),
        .mode = .simple,
    };
    link.applyAllPlatformLinks(cli_tests_artifact.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends);
    if (is_blocked_darwin) {
        cli_tests_artifact.use_llvm = true;
    }
    tui_tests_step = b.step("tui-tests", "Run TUI and CLI unit tests");
    if (is_blocked_darwin) {
        tui_tests_step.?.dependOn(&cli_tests_artifact.step);
    } else {
        const run_cli_tests = b.addRunArtifact(cli_tests_artifact);
        run_cli_tests.skip_foreign_checks = true;
        tui_tests_step.?.dependOn(&run_cli_tests.step);
    }

    if (targets.pathExists(b, "tools/cli/launcher_tests_root.zig")) {
        const launcher_tests_mod = b.createModule(.{
            .root_source_file = b.path("tools/cli/launcher_tests_root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        launcher_tests_mod.addImport("abi", abi_module);

        const launcher_tests = b.addTest(.{
            .root_module = launcher_tests_mod,
        });
        if (is_blocked_darwin) {
            launcher_tests.use_llvm = true;
        }

        launcher_tests_step = b.step("launcher-tests", "Run focused launcher and shell editor tests");
        if (is_blocked_darwin) {
            launcher_tests_step.?.dependOn(&launcher_tests.step);
        } else {
            const run_launcher_tests = b.addRunArtifact(launcher_tests);
            run_launcher_tests.skip_foreign_checks = true;
            launcher_tests_step.?.dependOn(&run_launcher_tests.step);
        }
    }

    // ── Lint / format ───────────────────────────────────────────────────
    const fmt_paths = &.{ "build.zig", "build", "src", "tools", "examples" };
    const lint_fmt = b.addFmt(.{ .paths = fmt_paths, .check = true });
    b.step("lint", "Check code formatting").dependOn(&lint_fmt.step);
    const fix_fmt = b.addFmt(.{ .paths = fmt_paths, .check = false });
    b.step("fix", "Format source files in place").dependOn(&fix_fmt.step);

    // ── Tests ───────────────────────────────────────────────────────────
    var test_step: ?*std.Build.Step = null;
    var typecheck_step: ?*std.Build.Step = null;
    var wdbx_fast_tests_step: ?*std.Build.Step = null;
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
        if (is_blocked_darwin) {
            tests.use_llvm = true;
        }
        typecheck_step = b.step("typecheck", "Compile tests without running");
        typecheck_step.?.dependOn(&tests.step);

        if (targets.pathExists(b, "src/core/database/wdbx.zig")) {
            const database_neural_mod = b.createModule(.{
                .root_source_file = b.path("src/core/database/wdbx.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            database_neural_mod.addImport("build_options", build_opts);
            database_neural_mod.addImport("shared_services", shared_services_module);
            const neural_wdbx_tests = b.addTest(.{
                .root_module = database_neural_mod,
            });
            if (is_blocked_darwin) {
                neural_wdbx_tests.use_llvm = true;
            }
            typecheck_step.?.dependOn(&neural_wdbx_tests.step);
        }

        if (targets.pathExists(b, "src/core/database_fast_tests_root.zig")) {
            const database_fast_tests_mod = b.createModule(.{
                .root_source_file = b.path("src/core/database_fast_tests_root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            database_fast_tests_mod.addImport("build_options", build_opts);
            database_fast_tests_mod.addImport("shared_services", shared_services_module);

            const database_fast_tests = b.addTest(.{
                .root_module = database_fast_tests_mod,
            });
            link.applyAllPlatformLinks(database_fast_tests.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends);
            if (is_blocked_darwin) {
                database_fast_tests.use_llvm = true;
            }

            wdbx_fast_tests_step = b.step("database-fast-tests", "Run focused database adapter tests (typecheck-only on blocked Darwin)");
            if (is_blocked_darwin) {
                wdbx_fast_tests_step.?.dependOn(&database_fast_tests.step);
            } else {
                const run_database_fast_tests = b.addRunArtifact(database_fast_tests);
                run_database_fast_tests.skip_foreign_checks = true;
                wdbx_fast_tests_step.?.dependOn(&run_database_fast_tests.step);
            }
            typecheck_step.?.dependOn(&database_fast_tests.step);
        }

        test_step = b.step("test", "Run unit tests");
        if (is_blocked_darwin) {
            // On blocked Darwin, "test" becomes typecheck-only (no linking/running)
            test_step.?.dependOn(&tests.step);
        } else {
            const run_tests = b.addRunArtifact(tests);
            run_tests.skip_foreign_checks = true;
            test_step.?.dependOn(&run_tests.step);
        }
    }

    // ── Feature tests (manifest-driven) ─────────────────────────────────
    const feature_tests_step = test_discovery.addFeatureTests(b, options, build_opts, abi_module, target, optimize);

    // ── Flag validation matrix ──────────────────────────────────────────
    const validate_flags_step = flags.addFlagValidation(b, target, optimize);

    // ── Import rule check ───────────────────────────────────────────────
    const import_check_step = b.step("check-imports", "Verify no @import(\"abi\") in feature modules");
    import_check_step.dependOn(addHostScriptStep(
        b,
        "abi-check-import-rules",
        "tools/scripts/check_import_rules.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
    ));

    // ── Consistency checks ──────────────────────────────────────────────
    const toolchain_doctor_step = b.step("toolchain-doctor", "Diagnose local Zig PATH/version drift against repository pin");
    toolchain_doctor_step.dependOn(addHostScriptStep(b, "abi-toolchain-doctor", "tools/scripts/toolchain_doctor.zig", target, optimize, &.{}, &.{.{ .name = "util", .module = util_module }}));

    const check_zig_version_step = b.step("check-zig-version", "Verify Zig version consistency");
    check_zig_version_step.dependOn(addHostScriptStep(
        b,
        "abi-check-zig-version-consistency",
        "tools/scripts/check_zig_version_consistency.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
    ));

    const check_test_baseline_step = b.step("check-test-baseline", "Verify test baseline consistency");
    check_test_baseline_step.dependOn(addHostScriptStep(
        b,
        "abi-check-test-baseline-consistency",
        "tools/scripts/check_test_baseline_consistency.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
    ));

    const check_zig_016_patterns_step = b.step("check-zig-016-patterns", "Verify Zig 0.16 conformance patterns");
    check_zig_016_patterns_step.dependOn(addHostScriptStep(
        b,
        "abi-check-zig-016-patterns",
        "tools/scripts/check_zig_016_patterns.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
    ));

    const check_feature_catalog_step = b.step("check-feature-catalog", "Verify feature catalog consistency");
    check_feature_catalog_step.dependOn(addHostScriptStep(
        b,
        "abi-check-feature-catalog",
        "tools/scripts/check_feature_catalog.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
    ));

    var check_gpu_policy_step: ?*std.Build.Step = null;
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
        if (is_blocked_darwin) {
            check_gpu_policy_exe.use_llvm = true;
        }
        check_gpu_policy_step = b.step("check-gpu-policy", "Verify GPU policy consistency");
        check_gpu_policy_step.?.dependOn(
            if (is_blocked_darwin) &check_gpu_policy_exe.step else &b.addRunArtifact(check_gpu_policy_exe).step,
        );
    }

    const ralph_gate_step = b.step("ralph-gate", "Require live Ralph scoring report and threshold pass");
    ralph_gate_step.dependOn(addHostScriptStep(b, "abi-check-ralph-gate", "tools/scripts/check_ralph_gate.zig", target, optimize, &.{}, &.{.{ .name = "util", .module = util_module }}));

    const workflow_contract_step = b.step("check-workflow-orchestration", "Advisory workflow-orchestration contract checks");
    workflow_contract_step.dependOn(addHostScriptStep(
        b,
        "abi-check-workflow-orchestration",
        "tools/scripts/check_workflow_orchestration.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
    ));

    const workflow_contract_strict_step = b.step("check-workflow-orchestration-strict", "Strict workflow-orchestration contract checks");
    workflow_contract_strict_step.dependOn(addHostScriptStep(
        b,
        "abi-check-workflow-orchestration-strict",
        "tools/scripts/check_workflow_orchestration.zig",
        target,
        optimize,
        &.{"--strict"},
        &.{.{ .name = "util", .module = util_module }},
    ));

    // ── Zig bootstrap steps ──────────────────────────────────────────────
    _ = cel.addZigBootstrapCheckStep(b);
    _ = cel.addZigBootstrapBuildStep(b);
    _ = cel.addZigBootstrapStatusStep(b);
    _ = cel.addZigBootstrapVerifyStep(b);
    _ = cel.addCelCheckStep(b);
    _ = cel.addCelBuildStep(b);
    _ = cel.addCelStatusStep(b);
    _ = cel.addCelVerifyStep(b);

    const zig_bootstrap_doctor_step = b.step("zig-bootstrap-doctor", "Run Zig bootstrap diagnostics and remediation");
    zig_bootstrap_doctor_step.dependOn(addHostScriptStep(b, "abi-zig-bootstrap-doctor", "tools/scripts/cel_doctor.zig", target, optimize, &.{}, &.{.{ .name = "util", .module = util_module }}));

    const cel_doctor_step = b.step("cel-doctor", "Deprecated alias for zig-bootstrap-doctor");
    cel_doctor_step.dependOn(addHostScriptStep(b, "abi-cel-doctor", "tools/scripts/cel_doctor.zig", target, optimize, &.{}, &.{.{ .name = "util", .module = util_module }}));

    // ── CLI DSL registry/codegen ───────────────────────────────────────
    const generate_cli_registry_step = b.step("generate-cli-registry", "Generate CLI registry artifact in build cache");
    generate_cli_registry_step.dependOn(addHostScriptStep(
        b,
        "abi-generate-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
        &.{ "--output", ".zig-cache/abi/generated/cli_registry.zig" },
        &.{.{ .name = "util", .module = util_module }},
    ));

    const refresh_cli_registry_step = b.step("refresh-cli-registry", "Refresh tracked CLI registry snapshot");
    refresh_cli_registry_step.dependOn(addHostScriptStep(
        b,
        "abi-refresh-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
        &.{"--snapshot"},
        &.{.{ .name = "util", .module = util_module }},
    ));

    const check_cli_registry_step = b.step("check-cli-registry", "Check CLI registry snapshot determinism");
    check_cli_registry_step.dependOn(addHostScriptStep(
        b,
        "abi-check-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
        &.{ "--check", "--snapshot" },
        &.{.{ .name = "util", .module = util_module }},
    ));

    const check_cli_dsl_consistency_step = b.step("check-cli-dsl-consistency", "Verify CLI/TUI DSL organization contracts");
    check_cli_dsl_consistency_step.dependOn(addHostScriptStep(
        b,
        "abi-check-cli-dsl-consistency",
        "tools/scripts/check_cli_dsl_consistency.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
    ));

    // ── Full check ──────────────────────────────────────────────────────
    const full_check_step = b.step("full-check", "Run the local confidence gate across deterministic leaf checks");
    full_check_step.dependOn(&lint_fmt.step);
    if (typecheck_step) |step| full_check_step.dependOn(step);
    if (test_step) |ts| full_check_step.dependOn(ts);
    full_check_step.dependOn(cli_tests_step);
    full_check_step.dependOn(validate_flags_step);
    full_check_step.dependOn(import_check_step);
    full_check_step.dependOn(check_zig_version_step);
    full_check_step.dependOn(check_test_baseline_step);
    full_check_step.dependOn(check_zig_016_patterns_step);
    full_check_step.dependOn(check_feature_catalog_step);
    if (check_gpu_policy_step) |step| full_check_step.dependOn(step);
    full_check_step.dependOn(check_cli_registry_step);
    full_check_step.dependOn(check_cli_dsl_consistency_step);
    full_check_step.dependOn(workflow_contract_strict_step);
    if (tui_tests_step) |step| full_check_step.dependOn(step);
    if (launcher_tests_step) |step| full_check_step.dependOn(step);
    if (wdbx_fast_tests_step) |step| full_check_step.dependOn(step);

    var check_docs_step: ?*std.Build.Step = null;

    // ── Documentation ───────────────────────────────────────────────────
    if (targets.pathExists(b, "tools/gendocs/main.zig")) {
        const gendocs_module = b.createModule(.{
            .root_source_file = b.path("tools/gendocs/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        gendocs_module.addImport("abi", abi_module);
        gendocs_module.addImport("cli_root", cli_root_mod);

        if (targets.pathExists(b, "build/gendocs_tests_root.zig")) {
            const gendocs_source_cli_module = b.createModule(.{
                .root_source_file = b.path("tools/gendocs/source_cli.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            gendocs_source_cli_module.addImport("cli_root", cli_root_mod);

            const gendocs_tests_root = b.createModule(.{
                .root_source_file = b.path("build/gendocs_tests_root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            gendocs_tests_root.addImport("gendocs_source_cli", gendocs_source_cli_module);

            const gendocs_source_tests = b.addTest(.{
                .root_module = gendocs_tests_root,
            });
            if (is_blocked_darwin) {
                gendocs_source_tests.use_llvm = true;
            }

            gendocs_source_tests_step = b.step("gendocs-source-tests", "Run focused gendocs CLI source discovery tests");
            if (is_blocked_darwin) {
                gendocs_source_tests_step.?.dependOn(&gendocs_source_tests.step);
            } else {
                const run_gendocs_source_tests = b.addRunArtifact(gendocs_source_tests);
                run_gendocs_source_tests.skip_foreign_checks = true;
                gendocs_source_tests_step.?.dependOn(&run_gendocs_source_tests.step);
            }
            full_check_step.dependOn(gendocs_source_tests_step.?);
        }

        const gendocs = b.addExecutable(.{
            .name = "gendocs",
            .root_module = gendocs_module,
        });
        if (is_blocked_darwin) {
            gendocs.use_llvm = true;
        }
        if (is_blocked_darwin) {
            // On blocked Darwin, gendocs/check-docs become compile-only
            b.step("gendocs", "Generate docs (compile-only on blocked Darwin)").dependOn(&gendocs.step);
            const docs_check = b.step("check-docs", "Validate docs (compile-only on blocked Darwin)");
            docs_check.dependOn(&gendocs.step);
            check_docs_step = docs_check;
            full_check_step.dependOn(docs_check);
        } else {
            const run_gendocs = b.addRunArtifact(gendocs);
            if (b.args) |args| run_gendocs.addArgs(args);
            b.step("gendocs", "Generate docs/api, docs/_docs, docs/plans, and docs/api-app").dependOn(&run_gendocs.step);

            const run_check_docs = b.addRunArtifact(gendocs);
            run_check_docs.addArg("--check");
            run_check_docs.addArg("--untracked-md");
            const docs_check = b.step("check-docs", "Validate docs generator determinism and output policy");
            docs_check.dependOn(&run_check_docs.step);
            check_docs_step = docs_check;
            full_check_step.dependOn(docs_check);
        }
    }

    // ── Profile build ───────────────────────────────────────────────────
    var profile_opts = options;
    profile_opts.feat_profiling = true;
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
    if (is_blocked_darwin) {
        profile_exe.use_llvm = true;
        b.step("profile", "Build with performance profiling (compile-only on blocked Darwin)").dependOn(&profile_exe.step);
    } else {
        b.installArtifact(profile_exe);
        b.step("profile", "Build with performance profiling").dependOn(b.getInstallStep());
    }

    // ── Mobile ──────────────────────────────────────────────────────────
    _ = mobile.addMobileBuild(b, options, optimize);

    // ── C Library ────────────────────────────────────────────────────────
    const c_bindings_src = "src/bindings/c/src/abi_c.zig";
    const c_bindings_header = "src/bindings/c/include/abi.h";
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
    if (targets.pathExists(b, "tools/scripts/check_perf.zig")) {
        const check_perf_exe = b.addExecutable(.{
            .name = "abi-check-perf",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/scripts/check_perf.zig"),
                .target = target,
                .optimize = .ReleaseSafe,
            }),
        });
        if (is_blocked_darwin) {
            check_perf_exe.use_llvm = true;
        }
        b.step("check-perf", "Build performance verification tool (pipe benchmark JSON to run)")
            .dependOn(if (is_blocked_darwin) &check_perf_exe.step else &b.addInstallArtifact(check_perf_exe, .{}).step);
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
        cross_opts.feat_mobile = false;
        if (ct.os == .wasi or ct.os == .freestanding or ct.os == .emscripten) {
            cross_opts.feat_database = false;
            cross_opts.feat_network = false;
            cross_opts.feat_gpu = false;
            cross_opts.feat_profiling = false;
            cross_opts.feat_web = false;
            cross_opts.feat_cloud = false;
            cross_opts.feat_storage = false;
            cross_opts.gpu_backends = &.{};
        } else {
            cross_opts.gpu_backends = &.{.stdgpu};
        }
        const cross_build_opts = modules.createBuildOptionsModule(b, cross_opts);
        const cross_shared_services = modules.createSharedServicesModule(b, cross_build_opts, cross_target, optimize);
        const cross_core_module = modules.createCoreModule(b, cross_target, optimize, cross_build_opts);
        const cross_abi_mod = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = cross_target,
            .optimize = optimize,
        });
        modules.wireAbiImports(cross_abi_mod, cross_build_opts, cross_shared_services, cross_core_module);
        const cross_lib = b.addLibrary(.{
            .name = "cross-" ++ ct.name,
            .root_module = cross_abi_mod,
            .linkage = .static,
        });
        cross_check_step.dependOn(&cross_lib.step);
    }

    // ── V3 Refactored Modules ─────────────────────────────────────────
    // New flat module structure: root.zig → core/database/, personas/, inference/, api_server/

    // Helper: create a v3 root module with package runtime imports wired in.
    const v3_root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    modules.wireAbiImports(v3_root_mod, build_opts, shared_services_module, core_module);

    // V3 Static library
    const v3_lib = b.addLibrary(.{
        .name = "abi-v3",
        .root_module = v3_root_mod,
        .linkage = .static,
    });
    b.step("v3-lib", "Build v3 static library").dependOn(&b.addInstallArtifact(v3_lib, .{}).step);

    // V3 Server executable
    const v3_server_mod = b.createModule(.{
        .root_source_file = b.path("tools/server/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    v3_server_mod.addImport("abi", abi_module);

    const v3_server = b.addExecutable(.{
        .name = "abi-server",
        .root_module = v3_server_mod,
    });
    if (is_blocked_darwin) {
        v3_server.use_llvm = true;
    }
    b.step("v3-server", "Build v3 server executable").dependOn(
        if (is_blocked_darwin) &v3_server.step else &b.addInstallArtifact(v3_server, .{}).step,
    );

    // V3 Tests
    const v3_test_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    modules.wireAbiImports(v3_test_mod, build_opts, shared_services_module, core_module);

    const v3_tests = b.addTest(.{
        .root_module = v3_test_mod,
    });
    if (is_blocked_darwin) {
        v3_tests.use_llvm = true;
    }
    const v3_test_step = b.step("v3-test", "Run v3 module tests");
    v3_test_step.dependOn(if (is_blocked_darwin) &v3_tests.step else &b.addRunArtifact(v3_tests).step);

    // ── Verify-all ──────────────────────────────────────────────────────
    const gate_hardening_step = b.step("gate-hardening", "Run deterministic gate hardening checks");
    gate_hardening_step.dependOn(toolchain_doctor_step);
    if (typecheck_step) |step| gate_hardening_step.dependOn(step);
    if (test_step) |step| gate_hardening_step.dependOn(step);
    gate_hardening_step.dependOn(cli_tests_step);
    if (tui_tests_step) |step| gate_hardening_step.dependOn(step);
    gate_hardening_step.dependOn(validate_flags_step);
    gate_hardening_step.dependOn(import_check_step);
    gate_hardening_step.dependOn(check_zig_version_step);
    gate_hardening_step.dependOn(check_test_baseline_step);
    gate_hardening_step.dependOn(check_zig_016_patterns_step);
    gate_hardening_step.dependOn(check_feature_catalog_step);
    if (check_gpu_policy_step) |step| gate_hardening_step.dependOn(step);
    gate_hardening_step.dependOn(check_cli_dsl_consistency_step);
    gate_hardening_step.dependOn(check_cli_registry_step);
    if (check_docs_step) |step| gate_hardening_step.dependOn(step);
    gate_hardening_step.dependOn(workflow_contract_strict_step);

    const verify_all_step = b.step("verify-all", "Run the superset validation gate across all deterministic leaf checks");
    verify_all_step.dependOn(&lint_fmt.step);
    if (typecheck_step) |step| verify_all_step.dependOn(step);
    if (test_step) |step| verify_all_step.dependOn(step);
    verify_all_step.dependOn(cli_tests_step);
    if (tui_tests_step) |step| verify_all_step.dependOn(step);
    verify_all_step.dependOn(validate_flags_step);
    verify_all_step.dependOn(import_check_step);
    verify_all_step.dependOn(check_zig_version_step);
    verify_all_step.dependOn(check_test_baseline_step);
    verify_all_step.dependOn(check_zig_016_patterns_step);
    verify_all_step.dependOn(check_feature_catalog_step);
    if (check_gpu_policy_step) |step| verify_all_step.dependOn(step);
    verify_all_step.dependOn(check_cli_registry_step);
    verify_all_step.dependOn(check_cli_dsl_consistency_step);
    verify_all_step.dependOn(workflow_contract_strict_step);
    if (wdbx_fast_tests_step) |step| verify_all_step.dependOn(step);
    verify_all_step.dependOn(feature_tests_step);
    verify_all_step.dependOn(examples_step);
    if (check_wasm_step) |s| verify_all_step.dependOn(s);
    if (check_docs_step) |docs_step| verify_all_step.dependOn(docs_step);
    verify_all_step.dependOn(cross_check_step);
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Resolve the native build target, clamping macOS version when needed.
///
/// Zig 0.16-dev's linker does not support macOS 26+ (Tahoe).  When the
/// build host reports a version the toolchain cannot handle, we clamp the
/// deployment target to macOS 15.0 so the linker can resolve
/// libSystem.B.dylib and friends from the installed SDK.  An explicit
/// `-Dtarget=` from the user is never overridden.
fn resolveNativeTarget(b: *std.Build) std.Build.ResolvedTarget {
    var query = b.standardTargetOptionsQueryOnly(.{});

    // Only patch when no explicit target was provided (native build) and
    // the build host is macOS with a version newer than Zig supports.
    if (query.os_tag == null and builtin.os.tag == .macos) {
        const native_ver = builtin.os.version_range.semver;
        if (native_ver.min.major >= 26) {
            const clamped: std.Target.Query.OsVersion = .{
                .semver = .{ .major = 14, .minor = 0, .patch = 0 },
            };
            if (query.os_version_min == null) query.os_version_min = clamped;
            if (query.os_version_max == null) query.os_version_max = clamped;
        }
    }

    return b.resolveTargetQuery(query);
}

/// Get just the compile step from a script runner (for blocked Darwin where
/// the host cannot execute standalone Zig validation binaries reliably).
fn addScriptCompileOnly(
    b: *std.Build,
    name: []const u8,
    source: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const obj = b.addObject(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(source),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    obj.use_llvm = true;
    return &obj.step;
}

fn addHostScriptStep(
    b: *std.Build,
    name: []const u8,
    source: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    args: []const []const u8,
    deps: []const struct { name: []const u8, module: *std.Build.Module },
) *std.Build.Step {
    if (is_blocked_darwin) {
        const obj = b.addObject(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(source),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        for (deps) |dep| obj.root_module.addImport(dep.name, dep.module);
        obj.use_llvm = true;
        return &obj.step;
    }

    const exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(source),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    for (deps) |dep| exe.root_module.addImport(dep.name, dep.module);
    const run = b.addRunArtifact(exe);
    for (args) |arg| run.addArg(arg);
    return &run.step;
}

fn addValidationScriptStep(
    b: *std.Build,
    name: []const u8,
    source: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    args: []const []const u8,
) *std.Build.Step {
    return addHostScriptStep(b, name, source, target, optimize, args, &.{});
}
