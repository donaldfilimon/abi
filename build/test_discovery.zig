const std = @import("std");
const link = @import("link.zig");
const options_mod = @import("options.zig");

const BuildOptions = options_mod.BuildOptions;

/// Build the feature-tests step using the abi module directly.
///
/// Zig 0.16 enforces single-module file ownership: each .zig file can
/// belong to exactly one named module.  The previous per-entry module
/// approach created N separate modules, causing ownership conflicts
/// whenever entries shared files through their import graphs.
///
/// The fix: use the `abi` module as the test root.  All `test {}` blocks
/// reachable from `src/root.zig` are compiled and run.
/// The feature_test_manifest remains as documentation and for future use
/// by isolated test harnesses that can handle module ownership.
pub fn addFeatureTests(
    b: *std.Build,
    options: BuildOptions,
    _: *std.Build.Module, // build_opts — abi_module already has build_options wired
    abi_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    _: std.builtin.OptimizeMode, // optimize — abi_module already has optimize configured
) *std.Build.Step {
    const ft_step = b.step("feature-tests", "Run feature module inline tests");

    const feature_tests = b.addTest(.{
        .root_module = abi_module,
    });
    link.applyAllPlatformLinks(
        feature_tests.root_module,
        target.result.os.tag,
        options.gpu_metal(),
        options.gpu_backends,
    );
    const run_feature_tests = b.addRunArtifact(feature_tests);
    run_feature_tests.skip_foreign_checks = true;
    ft_step.dependOn(&run_feature_tests.step);

    return ft_step;
}
