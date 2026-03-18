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
