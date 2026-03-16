const std = @import("std");
const link = @import("link.zig");
const module_catalog = @import("module_catalog.zig");
const options_mod = @import("options.zig");

const BuildOptions = options_mod.BuildOptions;

pub const FeatureTestEntry = module_catalog.FeatureTestEntry;
pub const feature_test_manifest = module_catalog.feature_test_manifest;

fn renderFeatureTestRoot(allocator: std.mem.Allocator) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    try out.writer.writeAll(
        "//! Generated feature module test discovery root.\n" ++
            "//! Source of truth: build/module_catalog.zig:feature_test_manifest.\n\n" ++
            "const build_options = @import(\"build_options\");\n\n" ++
            "test {\n",
    );

    for (feature_test_manifest, 0..) |entry, idx| {
        const alias = try std.fmt.allocPrint(allocator, "ft_{d}", .{idx});
        defer allocator.free(alias);

        if (entry.flag) |flag| {
            try out.writer.print("    if (build_options.{s}) _ = @import(\"{s}\");\n", .{ flag, alias });
        } else {
            try out.writer.print("    _ = @import(\"{s}\");\n", .{alias});
        }
    }

    try out.writer.writeAll("}\n");
    return try out.toOwnedSlice();
}

pub fn addFeatureTests(
    b: *std.Build,
    options: BuildOptions,
    build_opts: *std.Build.Module,
    abi_module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step {
    const generated = b.addWriteFiles();
    const root_source = renderFeatureTestRoot(b.allocator) catch @panic("renderFeatureTestRoot failed");
    const root_path = generated.add("generated_feature_tests.zig", root_source);

    const feature_test_root = b.createModule(.{
        .root_source_file = root_path,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    feature_test_root.addImport("build_options", build_opts);

    for (feature_test_manifest, 0..) |entry, idx| {
        const entry_module = b.createModule(.{
            .root_source_file = b.path(entry.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        entry_module.addImport("abi", abi_module);
        entry_module.addImport("build_options", build_opts);
        feature_test_root.addImport(b.fmt("ft_{d}", .{idx}), entry_module);
    }

    const is_blocked_darwin = @import("builtin").os.tag == .macos and @import("builtin").os.version_range.semver.min.major >= 26;
    const ft_step = b.step("feature-tests", "Run feature module inline tests");

    if (is_blocked_darwin) {
        const feature_tests = b.addObject(.{
            .name = "feature_tests",
            .root_module = feature_test_root,
        });
        feature_tests.use_llvm = true;
        ft_step.dependOn(&feature_tests.step);
    } else {
        const feature_tests = b.addTest(.{
            .root_module = feature_test_root,
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
    }

    return ft_step;
}
