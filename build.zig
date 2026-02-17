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
    if (target.result.os.tag == .macos and options.gpu_metal()) {
        exe.root_module.linkFramework("Metal", .{});
        exe.root_module.linkFramework("CoreML", .{});
        exe.root_module.linkFramework("MetalPerformanceShaders", .{});
        exe.root_module.linkFramework("Foundation", .{});
    }
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
        &.{ "gpu", "backends" },
        &.{ "gpu", "devices" },
        &.{ "gpu", "summary" },
        &.{ "gpu", "default" },
        &.{ "task", "list" },
        &.{ "task", "stats" },
        &.{ "config", "show" },
        &.{ "config", "validate" },
        &.{ "help", "llm" },
        &.{ "help", "gpu" },
        &.{ "help", "db" },
        &.{ "help", "train" },
        &.{ "help", "model" },
        &.{ "help", "config" },
        &.{ "help", "task" },
        &.{ "help", "network" },
        &.{ "help", "discord" },
        &.{ "help", "bench" },
        &.{ "help", "plugins" },
        &.{ "help", "completions" },
        &.{ "help", "multi-agent" },
        &.{ "help", "profile" },
        &.{ "help", "convert" },
        &.{ "help", "embed" },
        &.{ "help", "toolchain" },
        &.{ "help", "explore" },
        &.{ "help", "simd" },
        &.{ "help", "agent" },
        &.{ "help", "status" },
        &.{ "help", "mcp" },
        &.{ "help", "acp" },
        &.{ "help", "gpu-dashboard" },
        &.{ "help", "llm", "generate" },
        &.{ "help", "llm", "chat" },
        &.{ "help", "train", "run" },
        &.{ "help", "train", "llm" },
        &.{ "help", "db", "add" },
        &.{ "help", "bench", "simd" },
        &.{ "help", "discord", "commands" },
        &.{ "llm", "info" },
        &.{ "llm", "list" },
        &.{ "llm", "demo" },
        &.{ "llm", "demo", "--prompt", "Hello" },
        &.{ "train", "info" },
        &.{ "train", "auto" },
        &.{ "train", "auto", "--help" },
        &.{ "model", "list" },
        &.{ "network", "list" },
        &.{ "network", "status" },
        &.{ "discord", "status" },
        &.{ "discord", "commands", "list" },
        &.{ "plugins", "list" },
        &.{ "plugins", "info", "openai-connector" },
        &.{"bench"},
        &.{ "bench", "list" },
        &.{ "bench", "micro", "hash" },
        &.{ "bench", "micro", "alloc" },
        &.{ "completions", "bash" },
        &.{ "completions", "zsh" },
        &.{ "multi-agent", "info" },
        &.{ "multi-agent", "list" },
        &.{ "multi-agent", "status" },
        &.{ "toolchain", "status" },
        &.{ "mcp", "tools" },
        &.{ "acp", "card" },
        &.{ "acp", "serve", "--help" },
        // Nested help and subcommands (all must work)
        &.{ "help", "ralph" },
        &.{ "help", "gendocs" },
        &.{ "help", "db", "query" },
        &.{ "help", "db", "serve" },
        &.{ "help", "db", "backup" },
        &.{ "help", "db", "restore" },
        &.{ "help", "db", "optimize" },
        &.{ "help", "task", "add" },
        &.{ "help", "task", "edit" },
        &.{ "help", "ralph", "run" },
        &.{ "help", "ralph", "super" },
        &.{ "help", "ralph", "multi" },
        &.{ "help", "ralph", "skills" },
        &.{"version"},
        &.{"status"},
        &.{ "ralph", "help" },
        &.{ "ralph", "status" },
        &.{ "ralph", "skills" },
        &.{ "ralph", "gate", "--help" },
        &.{"gendocs"},
        &.{ "profile", "show" },
        &.{ "profile", "list" },
        &.{ "db", "add", "--help" },
        &.{ "db", "query", "--help" },
        &.{ "db", "optimize" },
        &.{ "db", "backup", "--help" },
        &.{ "db", "restore", "--help" },
        &.{ "db", "serve", "--help" },
        &.{ "help", "convert", "dataset" },
        &.{ "help", "convert", "model" },
        &.{ "help", "convert", "embeddings" },
        &.{ "help", "task", "edit" },
        &.{ "config", "init", "--help" },
        &.{ "help", "discord", "info" },
        &.{ "help", "discord", "guilds" },
        &.{ "llm", "list-local" },
        &.{ "llm", "serve", "--help" },
        &.{ "llm", "bench", "--help" },
        &.{ "llm", "download", "--help" },
        &.{ "train", "run", "--help" },
        &.{ "train", "new", "--help" },
        &.{ "train", "llm", "--help" },
        &.{ "train", "vision", "--help" },
        &.{ "train", "clip", "--help" },
        &.{ "train", "resume", "--help" },
        &.{ "train", "monitor", "--help" },
        &.{ "train", "generate-data", "--help" },
        &.{ "bench", "quick" },
        &.{ "bench", "simd" },
        &.{ "bench", "micro", "noop" },
        &.{ "bench", "micro", "parse" },
        &.{ "gpu", "list" },
        &.{ "agent", "--help" },
        &.{ "tui", "--help" },
        &.{ "embed", "--help" },
        &.{ "explore", "--help" },
        &.{ "model", "path" },
        &.{ "model", "info", "--help" },
        &.{ "plugins", "search" },
        &.{ "help", "plugins", "enable" },
        &.{ "help", "plugins", "disable" },
        &.{ "toolchain", "path" },
        &.{ "completions", "fish" },
        &.{ "completions", "powershell" },
        &.{ "multi-agent", "run", "--help" },
        &.{ "multi-agent", "create", "--help" },
        // Aliases
        &.{"info"},
        &.{ "ls", "stats" },
    };
    for (cli_commands) |args| {
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.addArgs(args);
        cli_tests_step.dependOn(&run_cmd.step);
    }

    // ── Lint ─────────────────────────────────────────────────────────────
    const fmt_paths = &.{
        "build.zig",
        "build",
        "src",
        "tools",
        "examples",
        "benchmarks",
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
        if (target.result.os.tag == .macos and options.gpu_metal()) {
            tests.root_module.linkFramework("Metal", .{});
            tests.root_module.linkFramework("CoreML", .{});
            tests.root_module.linkFramework("MetalPerformanceShaders", .{});
            tests.root_module.linkFramework("Foundation", .{});
        }
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
    const check_features = b.addSystemCommand(&.{ "bash", "scripts/check_feature_catalog_consistency.sh" });
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

    // ── Benchmarks ───────────────────────────────────────────────────────
    const bench_all_step = b.step("bench-all", "Run all benchmark suites");
    targets.buildTargets(b, &targets.benchmark_targets, abi_module, build_opts, target, optimize, bench_all_step, true);
    const bench_step = b.step("bench", "Alias for bench-all benchmark suites");
    bench_step.dependOn(bench_all_step);

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
        const profile_mod = profile_exe.root_module;
        profile_mod.addImport("abi", abi_profile);
        profile_mod.addImport("cli", modules.createCliModule(b, abi_profile, target, optimize));
        profile_mod.strip = false;
        profile_mod.omit_frame_pointer = false;
        if (target.result.os.tag == .macos and options.gpu_metal()) {
            profile_mod.linkFramework("Metal", .{});
            profile_mod.linkFramework("CoreML", .{});
            profile_mod.linkFramework("MetalPerformanceShaders", .{});
            profile_mod.linkFramework("Foundation", .{});
        }
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
    const verify_all_step = b.step("verify-all", "full-check + consistency checks + examples + bench-all + check-wasm + ralph-gate");
    verify_all_step.dependOn(full_check_step);
    verify_all_step.dependOn(consistency_step);
    verify_all_step.dependOn(examples_step);
    verify_all_step.dependOn(bench_all_step);
    verify_all_step.dependOn(ralph_gate_step);
    if (check_wasm_step) |s| verify_all_step.dependOn(s);
}
