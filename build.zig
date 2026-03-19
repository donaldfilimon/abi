//! ABI build root. Zig 0.16 Build API: root_module, createModule, b.path (LazyPath).
const std = @import("std");

const cli_tests = @import("build/cli_tests.zig");
const darwin = @import("build/darwin.zig");
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

pub fn build(b: *std.Build) void {
    const ctx = darwin.initDarwinCtx(b);
    if (ctx.is_blocked) {
        std.log.warn(
            \\
            \\  ╔══════════════════════════════════════════════════════════╗
            \\  ║  Darwin 25+ detected — stock Zig linker is blocked.    ║
            \\  ║                                                        ║
            \\  ║  Use:  ./tools/scripts/run_build.sh <args>             ║
            \\  ║    or: PATH=tools/scripts:$PATH zig build <args>       ║
            \\  ║    or: bootstrap host Zig via bootstrap_host_zig.sh    ║
            \\  ╚══════════════════════════════════════════════════════════╝
            \\
        , .{});
    }
    const target = darwin.resolveNativeTarget(b);
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
    if (ctx.is_blocked) {
        options.feat_database = false;
        options.feat_ai = false;
        options.feat_explore = false;
        options.feat_llm = false;
        options.feat_vision = false;
        options.feat_training = false;
        options.feat_reasoning = false;
    }
    options_mod.validateOptions(options);

    // ── Core modules ────────────────────────────────────────────────────
    const build_opts = modules.createBuildOptionsModule(b, options);

    const util_module = b.addModule("util", .{
        .root_source_file = b.path("tools/scripts/util.zig"),
        .target = target,
        .optimize = optimize,
    });

    const toolchain_support_module = b.addModule("toolchain_support", .{
        .root_source_file = b.path("tools/scripts/toolchain_support.zig"),
        .target = target,
        .optimize = optimize,
    });

    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    modules.wireAbiImports(abi_module, build_opts);

    // ── CLI executable ──────────────────────────────────────────────────
    const cli_module = b.createModule(.{
        .root_source_file = b.path("tools/cli/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const cli_artifact = darwin.addExeOrObject(b, "abi", cli_module, ctx);
    cli_artifact.compile.root_module.addImport("abi", abi_module);
    cli_artifact.compile.root_module.addImport("cli", modules.createCliModule(b, abi_module, toolchain_support_module, target, optimize));
    targets.applyPerformanceTweaks(cli_artifact.compile, optimize);
    link.applyAllPlatformLinks(cli_artifact.compile.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends, ctx.is_blocked);

    const run_cli = darwin.addRunStep(b, cli_artifact, "abi_linked", ctx);
    if (b.args) |args| run_cli.addArgs(args);
    b.step("run", "Run the ABI CLI").dependOn(&run_cli.step);

    const run_editor = darwin.addRunStep(b, cli_artifact, "abi_editor_linked", ctx);
    run_editor.addArg("ui");
    run_editor.addArg("editor");
    if (b.args) |args| run_editor.addArgs(args);
    b.step("editor", "Run the inline CLI TUI editor").dependOn(&run_editor.step);

    if (!ctx.is_blocked) {
        b.installArtifact(cli_artifact.compile);
    }

    // ── Examples (table-driven) ─────────────────────────────────────────
    const examples_step = b.step("examples", "Build all examples");
    targets.buildTargets(b, &targets.example_targets, abi_module, build_opts, target, optimize, ctx, examples_step, false);

    // ── CLI smoke tests ─────────────────────────────────────────────────
    const cli_tests_step = cli_tests.addCliTests(
        b,
        cli_artifact.compile,
        abi_module,
        toolchain_support_module,
        target,
        optimize,
        ctx,
    );

    // ── CLI full integration tests (matrix manifest) ────────────────────
    _ = cli_tests.addCliTestsFull(
        b,
        cli_artifact.compile,
        target,
        optimize,
        ctx,
    );

    // ── TUI / CLI unit tests ───────────────────────────────────────────
    var tui_tests_step: ?*std.Build.Step = null;
    var gendocs_source_tests_step: ?*std.Build.Step = null;
    var launcher_tests_step: ?*std.Build.Step = null;
    const cli_root_mod = modules.createCliModule(b, abi_module, toolchain_support_module, target, optimize);

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
    link.applyAllPlatformLinks(cli_tests_artifact.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends, ctx.is_blocked);
    darwin.enableLlvm(cli_tests_artifact, ctx);
    tui_tests_step = b.step("tui-tests", "Run TUI and CLI unit tests");
    tui_tests_step.?.dependOn(darwin.addTestRunStep(b, cli_tests_artifact, ctx));

    if (targets.pathExists(b, "tools/cli/launcher_tests_root.zig")) {
        const launcher_tests_mod = b.createModule(.{
            .root_source_file = b.path("tools/cli/launcher_tests_root.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        launcher_tests_mod.addImport("abi", abi_module);
        launcher_tests_mod.addImport("toolchain_support", toolchain_support_module);

        const launcher_tests = b.addTest(.{ .root_module = launcher_tests_mod });
        darwin.enableLlvm(launcher_tests, ctx);
        launcher_tests_step = b.step("launcher-tests", "Run focused launcher and shell editor tests");
        launcher_tests_step.?.dependOn(darwin.addTestRunStep(b, launcher_tests, ctx));
    }

    // ── Lint / format ───────────────────────────────────────────────────
    const fmt_paths = &.{ "build.zig", "build", "src", "tools", "examples", "tests", "bindings", "lang" };
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
        link.applyAllPlatformLinks(tests.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends, ctx.is_blocked);
        darwin.enableLlvm(tests, ctx);
        typecheck_step = b.step("typecheck", "Compile tests without running");
        typecheck_step.?.dependOn(&tests.step);

        if (targets.pathExists(b, "src/database_wdbx_tests_root.zig")) {
            const database_neural_mod = b.createModule(.{
                .root_source_file = b.path("src/database_wdbx_tests_root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            database_neural_mod.addImport("build_options", build_opts);
            const neural_wdbx_tests = b.addTest(.{ .root_module = database_neural_mod });
            darwin.enableLlvm(neural_wdbx_tests, ctx);
            typecheck_step.?.dependOn(&neural_wdbx_tests.step);
        }

        if (targets.pathExists(b, "src/database_fast_tests_root.zig")) {
            const database_fast_tests_mod = b.createModule(.{
                .root_source_file = b.path("src/database_fast_tests_root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            database_fast_tests_mod.addImport("build_options", build_opts);

            const database_fast_tests = b.addTest(.{ .root_module = database_fast_tests_mod });
            link.applyAllPlatformLinks(database_fast_tests.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends, ctx.is_blocked);
            darwin.enableLlvm(database_fast_tests, ctx);

            wdbx_fast_tests_step = b.step("database-fast-tests", "Run focused database adapter tests (typecheck-only on blocked Darwin)");
            wdbx_fast_tests_step.?.dependOn(darwin.addTestRunStep(b, database_fast_tests, ctx));
            typecheck_step.?.dependOn(&database_fast_tests.step);
        }

        test_step = b.step("test", "Run unit tests");
        test_step.?.dependOn(darwin.addTestRunStep(b, tests, ctx));
    }

    // ── Feature tests (manifest-driven) ─────────────────────────────────
    const feature_tests_step = test_discovery.addFeatureTests(b, options, build_opts, abi_module, target, optimize, ctx);

    // ── Stub parity check ─────────────────────────────────────────────
    const parity_mod = b.createModule(.{
        .root_source_file = b.path("src/feature_parity_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    parity_mod.addImport("build_options", build_opts);
    const parity_tests = b.addTest(.{ .root_module = parity_mod });
    darwin.enableLlvm(parity_tests, ctx);
    const check_stub_parity_step = b.step("check-stub-parity", "Verify mod/stub declaration parity across all features");
    check_stub_parity_step.dependOn(darwin.addTestRunStep(b, parity_tests, ctx));

    // ── Flag validation matrix ──────────────────────────────────────────
    const validate_flags_step = flags.addFlagValidation(b, target, optimize);

    // ── Import rule check ───────────────────────────────────────────────
    const import_check_step = b.step("check-imports", "Verify feature import rules and named-vs-file import hygiene");
    import_check_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-import-rules",
        "tools/scripts/check_import_rules.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    // ── Consistency checks ──────────────────────────────────────────────
    // Darwin pipeline self-test (only meaningful on blocked Darwin hosts)
    const darwin_self_test_step = b.step("darwin-self-test", "Validate Darwin 25+ relink pipeline (SDK, ld, compiler_rt)");
    if (ctx.is_blocked) {
        const self_test = b.addSystemCommand(&.{ "./tools/scripts/run_build.sh", "--self-test" });
        darwin_self_test_step.dependOn(&self_test.step);
    }

    const toolchain_doctor_step = b.step("toolchain-doctor", "Diagnose local Zig PATH/version drift against repository pin");
    toolchain_doctor_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-toolchain-doctor",
        "tools/scripts/toolchain_doctor.zig",
        target,
        optimize,
        &.{},
        &.{
            .{ .name = "util", .module = util_module },
            .{ .name = "toolchain_support", .module = toolchain_support_module },
        },
        ctx,
    ));

    const preflight_step = b.step("preflight", "Run integration-test preflight environment diagnostics");
    preflight_step.dependOn(darwin.addHostScriptStep(b, "abi-preflight", "tests/integration/preflight.zig", target, optimize, &.{}, &.{.{ .name = "util", .module = util_module }}, ctx));

    const check_zig_version_step = b.step("check-zig-version", "Verify Zig version consistency");
    check_zig_version_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-zig-version-consistency",
        "tools/scripts/check_zig_version_consistency.zig",
        target,
        optimize,
        &.{},
        &.{
            .{ .name = "util", .module = util_module },
            .{ .name = "toolchain_support", .module = toolchain_support_module },
        },
        ctx,
    ));

    const check_test_baseline_step = b.step("check-test-baseline", "Verify test baseline consistency");
    check_test_baseline_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-test-baseline-consistency",
        "tools/scripts/check_test_baseline_consistency.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    const check_test_coverage_step = b.step("check-test-coverage", "Detect orphaned test files not reachable from any test root");
    check_test_coverage_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-test-coverage",
        "tools/scripts/check_test_coverage.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    const check_zig_016_patterns_step = b.step("check-zig-016-patterns", "Verify Zig 0.16 conformance patterns");
    check_zig_016_patterns_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-zig-016-patterns",
        "tools/scripts/check_zig_016_patterns.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    const check_feature_catalog_step = b.step("check-feature-catalog", "Verify feature catalog consistency");
    check_feature_catalog_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-feature-catalog",
        "tools/scripts/check_feature_catalog.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
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
        const gpu_policy_artifact = darwin.addExeOrObject(b, "abi-check-gpu-policy-consistency", gpu_policy_check_module, ctx);
        check_gpu_policy_step = b.step("check-gpu-policy", "Verify GPU policy consistency");
        if (gpu_policy_artifact.is_object) {
            check_gpu_policy_step.?.dependOn(&gpu_policy_artifact.compile.step);
        } else {
            check_gpu_policy_step.?.dependOn(&b.addRunArtifact(gpu_policy_artifact.compile).step);
        }
    }

    const ralph_gate_step = b.step("ralph-gate", "Require live Ralph scoring report and threshold pass");
    ralph_gate_step.dependOn(darwin.addHostScriptStep(b, "abi-check-ralph-gate", "tools/scripts/check_ralph_gate.zig", target, optimize, &.{}, &.{.{ .name = "util", .module = util_module }}, ctx));

    const workflow_contract_step = b.step("check-workflow-orchestration", "Advisory workflow-orchestration contract checks");
    workflow_contract_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-workflow-orchestration",
        "tools/scripts/check_workflow_orchestration.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    const workflow_contract_strict_step = b.step("check-workflow-orchestration-strict", "Strict workflow-orchestration contract checks");
    workflow_contract_strict_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-workflow-orchestration-strict",
        "tools/scripts/check_workflow_orchestration.zig",
        target,
        optimize,
        &.{"--strict"},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    // ── CLI DSL registry/codegen ───────────────────────────────────────
    const generate_cli_registry_step = b.step("generate-cli-registry", "Generate CLI registry artifact in build cache");
    generate_cli_registry_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-generate-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
        &.{ "--output", ".zig-cache/abi/generated/cli_registry.zig" },
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    const refresh_cli_registry_step = b.step("refresh-cli-registry", "Refresh tracked CLI registry snapshot");
    refresh_cli_registry_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-refresh-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
        &.{"--snapshot"},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    const check_cli_registry_step = b.step("check-cli-registry", "Check CLI registry snapshot determinism");
    check_cli_registry_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-cli-registry",
        "tools/scripts/generate_cli_registry.zig",
        target,
        optimize,
        &.{ "--check", "--snapshot" },
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    const check_cli_dsl_consistency_step = b.step("check-cli-dsl-consistency", "Verify CLI/TUI DSL organization contracts");
    check_cli_dsl_consistency_step.dependOn(darwin.addHostScriptStep(
        b,
        "abi-check-cli-dsl-consistency",
        "tools/scripts/check_cli_dsl_consistency.zig",
        target,
        optimize,
        &.{},
        &.{.{ .name = "util", .module = util_module }},
        ctx,
    ));

    // ── Check aggregation ───────────────────────────────────────────────
    const core_checks = [_]*std.Build.Step{
        cli_tests_step,
        validate_flags_step,
        import_check_step,
        check_zig_version_step,
        check_test_baseline_step,
        check_zig_016_patterns_step,
        check_feature_catalog_step,
        check_cli_registry_step,
        check_cli_dsl_consistency_step,
        workflow_contract_strict_step,
        check_stub_parity_step,
    };
    const optional_checks = [_]?*std.Build.Step{
        typecheck_step,
        test_step,
        check_gpu_policy_step,
        tui_tests_step,
    };
    const extra_checks = [_]?*std.Build.Step{
        launcher_tests_step,
        wdbx_fast_tests_step,
    };

    // ── Full check ──────────────────────────────────────────────────────
    const full_check_step = b.step("full-check", "Run the local confidence gate across deterministic leaf checks");
    full_check_step.dependOn(&lint_fmt.step);
    wireChecks(full_check_step, &core_checks, &optional_checks);
    for (&extra_checks) |opt| if (opt) |s| full_check_step.dependOn(s);

    var check_docs_step: ?*std.Build.Step = null;

    // ── Documentation ───────────────────────────────────────────────────
    if (targets.pathExists(b, "tools/gendocs/main.zig")) {
        const module_catalog_mod = b.createModule(.{
            .root_source_file = b.path("build/module_catalog.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        const gendocs_module = b.createModule(.{
            .root_source_file = b.path("tools/gendocs/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        gendocs_module.addImport("abi", abi_module);
        gendocs_module.addImport("cli_root", cli_root_mod);
        gendocs_module.addImport("module_catalog", module_catalog_mod);

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

            const gendocs_source_tests = b.addTest(.{ .root_module = gendocs_tests_root });
            darwin.enableLlvm(gendocs_source_tests, ctx);

            gendocs_source_tests_step = b.step("gendocs-source-tests", "Run focused gendocs CLI source discovery tests");
            gendocs_source_tests_step.?.dependOn(darwin.addTestRunStep(b, gendocs_source_tests, ctx));
            full_check_step.dependOn(gendocs_source_tests_step.?);
        }

        const gendocs_artifact = darwin.addExeOrObject(b, "gendocs", gendocs_module, ctx);
        link.applyAllPlatformLinks(gendocs_artifact.compile.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends, ctx.is_blocked);

        const run_gendocs = darwin.addRunStep(b, gendocs_artifact, "gendocs_linked", ctx);
        if (b.args) |args| run_gendocs.addArgs(args);
        b.step("gendocs", "Generate docs/api, docs/_docs, docs/plans, and docs/api-app").dependOn(&run_gendocs.step);

        const run_check_docs = darwin.addRunStep(b, gendocs_artifact, "gendocs_check_linked", ctx);
        run_check_docs.addArg("--check");
        run_check_docs.addArg("--untracked-md");

        const docs_check = b.step("check-docs", "Validate docs generator determinism and output policy");
        docs_check.dependOn(&run_check_docs.step);
        check_docs_step = docs_check;
        full_check_step.dependOn(docs_check);
    }

    // ── Profile build ───────────────────────────────────────────────────
    var profile_opts = options;
    profile_opts.feat_profiling = true;
    const abi_profile = modules.createAbiModule(b, profile_opts, target, optimize);
    const profile_mod = b.createModule(.{
        .root_source_file = b.path("tools/cli/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libc = true,
    });
    const profile_artifact = darwin.addExeOrObject(b, "abi-profile", profile_mod, ctx);
    profile_artifact.compile.root_module.addImport("abi", abi_profile);
    profile_artifact.compile.root_module.addImport("cli", modules.createCliModule(b, abi_profile, toolchain_support_module, target, optimize));
    profile_artifact.compile.root_module.strip = false;
    profile_artifact.compile.root_module.omit_frame_pointer = false;
    link.applyAllPlatformLinks(profile_artifact.compile.root_module, target.result.os.tag, options.gpu_metal(), options.gpu_backends, ctx.is_blocked);
    if (ctx.is_blocked) {
        b.step("profile", "Build with performance profiling (compile-only on blocked Darwin)").dependOn(&profile_artifact.compile.step);
    } else {
        b.installArtifact(profile_artifact.compile);
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
    if (targets.pathExists(b, "tools/scripts/check_perf.zig")) {
        const perf_artifact = darwin.addExeOrObject(b, "abi-check-perf", b.createModule(.{
            .root_source_file = b.path("tools/scripts/check_perf.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
        }), ctx);
        b.step("check-perf", "Build performance verification tool (pipe benchmark JSON to run)")
            .dependOn(if (perf_artifact.is_object) &perf_artifact.compile.step else &b.addInstallArtifact(perf_artifact.compile, .{}).step);
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
        if (ct.os == .ios or ct.abi == .android) {
            cross_opts.feat_mobile = true;
        } else {
            cross_opts.feat_mobile = false;
        }
        if (ct.os == .wasi or ct.os == .freestanding or ct.os == .emscripten) {
            cross_opts.feat_database = false;
            cross_opts.feat_network = false;
            cross_opts.feat_gpu = false;
            cross_opts.feat_profiling = false;
            cross_opts.feat_web = false;
            cross_opts.feat_cloud = false;
            cross_opts.feat_storage = false;
            cross_opts.feat_lsp = false;
            cross_opts.feat_mcp = false;
            cross_opts.gpu_backends = &.{};
        } else {
            cross_opts.gpu_backends = &.{.stdgpu};
        }
        const cross_build_opts = modules.createBuildOptionsModule(b, cross_opts);
        const cross_abi_mod = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = cross_target,
            .optimize = optimize,
        });
        modules.wireAbiImports(cross_abi_mod, cross_build_opts);
        const cross_lib = b.addLibrary(.{
            .name = "cross-" ++ ct.name,
            .root_module = cross_abi_mod,
            .linkage = .static,
        });
        cross_check_step.dependOn(&cross_lib.step);
    }

    // ── Additional Build Targets ──────────────────────────────────────
    const static_root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    modules.wireAbiImports(static_root_mod, build_opts);

    const static_lib = b.addLibrary(.{
        .name = "abi-static",
        .root_module = static_root_mod,
        .linkage = .static,
    });
    b.step("static-lib", "Build static Zig library").dependOn(&b.addInstallArtifact(static_lib, .{}).step);

    // Server executable
    const server_artifact = darwin.addExeOrObject(b, "abi-server", b.createModule(.{
        .root_source_file = b.path("tools/server/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }), ctx);
    server_artifact.compile.root_module.addImport("abi", abi_module);
    b.step("server", "Build server executable").dependOn(
        if (server_artifact.is_object) &server_artifact.compile.step else &b.addInstallArtifact(server_artifact.compile, .{}).step,
    );

    // ── Gate hardening ────────────────────────────────────────────────
    const gate_hardening_step = b.step("gate-hardening", "Run deterministic gate hardening checks");
    gate_hardening_step.dependOn(toolchain_doctor_step);
    wireChecks(gate_hardening_step, &core_checks, &optional_checks);
    if (check_docs_step) |step| gate_hardening_step.dependOn(step);

    // ── Verify-all ──────────────────────────────────────────────────────
    const verify_all_step = b.step("verify-all", "Run the superset validation gate across all deterministic leaf checks");
    verify_all_step.dependOn(&lint_fmt.step);
    wireChecks(verify_all_step, &core_checks, &optional_checks);
    if (wdbx_fast_tests_step) |step| verify_all_step.dependOn(step);
    verify_all_step.dependOn(feature_tests_step);
    verify_all_step.dependOn(examples_step);
    if (check_wasm_step) |s| verify_all_step.dependOn(s);
    if (check_docs_step) |docs_step| verify_all_step.dependOn(docs_step);
    verify_all_step.dependOn(cross_check_step);
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Wire both core (required) and optional checks into a gate step.
fn wireChecks(
    gate: *std.Build.Step,
    core: []const *std.Build.Step,
    optional: []const ?*std.Build.Step,
) void {
    for (core) |s| gate.dependOn(s);
    for (optional) |opt| if (opt) |s| gate.dependOn(s);
}
