const std = @import("std");
const darwin = @import("darwin.zig");
const modules = @import("modules.zig");

pub fn addCliTests(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    abi_module: *std.Build.Module,
    toolchain_support_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ctx: darwin.DarwinCtx,
) *std.Build.Step {
    const cli_root = modules.createCliModule(b, abi_module, toolchain_support_module, target, optimize);

    const smoke_runner_mod = b.createModule(.{
        .root_source_file = b.path("build/cli_smoke_runner.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    smoke_runner_mod.addImport("cli_root", cli_root);

    const smoke_runner = b.addExecutable(.{
        .name = "abi-cli-smoke-runner",
        .root_module = smoke_runner_mod,
    });
    darwin.enableLlvm(smoke_runner, ctx);

    const run_smoke = b.addRunArtifact(smoke_runner);
    run_smoke.addArg("--bin");
    run_smoke.addArtifactArg(exe);
    run_smoke.skip_foreign_checks = true;

    const step = b.step("cli-tests", "Run descriptor-driven CLI smoke coverage");
    if (ctx.is_blocked) {
        const cli_main_check = b.addObject(.{
            .name = "abi-cli-main-typecheck",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/cli/main.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        cli_main_check.root_module.addImport("cli", modules.createCliModule(b, abi_module, toolchain_support_module, target, optimize));
        darwin.enableLlvm(cli_main_check, ctx);

        const smoke_runner_check = b.addObject(.{
            .name = "abi-cli-smoke-runner-typecheck",
            .root_module = smoke_runner_mod,
        });
        darwin.enableLlvm(smoke_runner_check, ctx);

        step.dependOn(&cli_main_check.step);
        step.dependOn(&smoke_runner_check.step);
    } else {
        step.dependOn(&run_smoke.step);
    }
    return step;
}

/// Register the `cli-tests-full` build step.
///
/// This step runs the exhaustive integration vectors from
/// `tests/integration/matrix_manifest.zig` (excluding TUI vectors that
/// need a PTY).  On Darwin degraded mode it falls back to typecheck-only.
pub fn addCliTestsFull(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ctx: darwin.DarwinCtx,
) *std.Build.Step {
    const matrix_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration/matrix_manifest.zig"),
        .target = target,
        .optimize = optimize,
    });

    const full_runner_mod = b.createModule(.{
        .root_source_file = b.path("build/cli_full_runner.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    full_runner_mod.addImport("matrix_manifest", matrix_mod);

    const full_runner = b.addExecutable(.{
        .name = "abi-cli-full-runner",
        .root_module = full_runner_mod,
    });
    darwin.enableLlvm(full_runner, ctx);

    const run_full = b.addRunArtifact(full_runner);
    run_full.addArg("--bin");
    run_full.addArtifactArg(exe);
    run_full.skip_foreign_checks = true;

    const step = b.step("cli-tests-full", "Run exhaustive CLI integration vectors from matrix manifest");
    if (ctx.is_blocked) {
        const full_runner_check = b.addObject(.{
            .name = "abi-cli-full-runner-typecheck",
            .root_module = full_runner_mod,
        });
        darwin.enableLlvm(full_runner_check, ctx);

        step.dependOn(&full_runner_check.step);
    } else {
        step.dependOn(&run_full.step);
    }
    return step;
}
