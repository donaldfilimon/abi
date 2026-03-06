const std = @import("std");

pub fn addCliTests(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    abi_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const cli_root = b.createModule(.{
        .root_source_file = b.path("tools/cli/mod.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    cli_root.addImport("abi", abi_module);

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

    const run_smoke = b.addRunArtifact(smoke_runner);
    run_smoke.addArg("--bin");
    run_smoke.addArtifactArg(exe);
    run_smoke.skip_foreign_checks = true;

    const step = b.step("cli-tests", "Run descriptor-driven CLI smoke coverage");
    step.dependOn(&run_smoke.step);
    return step;
}
