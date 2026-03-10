const std = @import("std");
const modules = @import("modules.zig");

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
    const is_blocked_darwin = @import("builtin").os.tag == .macos and @import("builtin").os.version_range.semver.min.major >= 26;
    if (is_blocked_darwin) {
        smoke_runner.use_llvm = true;
        // LLD has zero Mach-O support; Apple /usr/bin/ld used via run_build.sh
    }

    const run_smoke = b.addRunArtifact(smoke_runner);
    run_smoke.addArg("--bin");
    run_smoke.addArtifactArg(exe);
    run_smoke.skip_foreign_checks = true;

    const step = b.step("cli-tests", "Run descriptor-driven CLI smoke coverage");
    if (is_blocked_darwin) {
        const cli_main_check = b.addObject(.{
            .name = "abi-cli-main-typecheck",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/cli/main.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        });
        cli_main_check.root_module.addImport("cli", modules.createCliModule(b, abi_module, target, optimize));
        cli_main_check.use_llvm = true;

        const smoke_runner_check = b.addObject(.{
            .name = "abi-cli-smoke-runner-typecheck",
            .root_module = smoke_runner_mod,
        });
        smoke_runner_check.use_llvm = true;

        step.dependOn(&cli_main_check.step);
        step.dependOn(&smoke_runner_check.step);
    } else {
        step.dependOn(&run_smoke.step);
    }
    return step;
}
