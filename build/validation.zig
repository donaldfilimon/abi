const std = @import("std");
const builtin = @import("builtin");
const build_flags = @import("flags.zig");
const linking = @import("linking.zig");

pub const Context = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    flags: build_flags.FeatureFlags,
    build_options_module: *std.Build.Module,
    abi_module: *std.Build.Module,
    package_version: []const u8 = "0.1.0",
};

// Feature test lane definitions: each entry creates a "{display}-tests" build step
// wiring "src/{name}_mod_test.zig" (unit) + "test/integration/{integration_name}_test.zig".
// integration_name defaults to name when null.
const FeatureLane = struct { name: []const u8, display: []const u8, integration_name: ?[]const u8 = null };
const feature_lanes: []const FeatureLane = &.{
    .{ .name = "messaging", .display = "messaging" },
    .{ .name = "secrets", .display = "secrets" },
    .{ .name = "pitr", .display = "PITR" },
    .{ .name = "agents", .display = "agents" },
    .{ .name = "multi_agent", .display = "multi-agent" },
    .{ .name = "orchestration", .display = "orchestration" },
    .{ .name = "gateway", .display = "gateway" },
    .{ .name = "inference", .display = "inference" },
    .{ .name = "gpu", .display = "gpu" },
    .{ .name = "network", .display = "network" },
    .{ .name = "web", .display = "web" },
    .{ .name = "observability", .display = "observability" },
    .{ .name = "search", .display = "search" },
    .{ .name = "auth", .display = "auth" },
    .{ .name = "storage", .display = "storage" },
    .{ .name = "cloud", .display = "cloud" },
    .{ .name = "cache", .display = "cache" },
    .{ .name = "database", .display = "database" },
    .{ .name = "connectors", .display = "connectors", .integration_name = "connector" },
    .{ .name = "lsp", .display = "lsp" },
    .{ .name = "acp", .display = "acp" },
    .{ .name = "ha", .display = "ha" },
    .{ .name = "tasks", .display = "tasks" },
    .{ .name = "documents", .display = "documents" },
    .{ .name = "compute", .display = "compute" },
    .{ .name = "desktop", .display = "desktop" },
    .{ .name = "pipeline", .display = "pipeline" },
};

pub const Steps = struct {
    test_step: *std.Build.Step,
    check_parity_step: *std.Build.Step,
    feature_tests_step: *std.Build.Step,
    mcp_tests_step: *std.Build.Step,
    check_step: *std.Build.Step,
};

pub fn addSteps(ctx: Context) Steps {
    const test_step = ctx.b.step("test", "Run tests");

    const lib_tests = addTests(ctx.b, ctx.target, ctx.optimize, "src/root.zig", ctx.build_options_module, null);
    linking.linkIfDarwin(lib_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const run_lib_tests = ctx.b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_lib_tests.step);

    const integration_tests = addTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/mod.zig",
        ctx.build_options_module,
        ctx.abi_module,
    );
    linking.linkIfDarwin(integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const run_integration = ctx.b.addRunArtifact(integration_tests);
    test_step.dependOn(&run_integration.step);

    addFlagOverrideTestLane(ctx, "feat_tui", true, "tui-tests", "Run TUI tests with feat-tui=true");

    const parity_tests = addParityTests(ctx.b, ctx.target, ctx.optimize, ctx.build_options_module);
    linking.linkIfDarwin(parity_tests, .parity_test, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const run_parity = ctx.b.addRunArtifact(parity_tests);
    const check_parity_step = ctx.b.step("check-parity", "Verify mod/stub declaration parity");
    check_parity_step.dependOn(&run_parity.step);

    const feature_tests_step = ctx.b.step("feature-tests", "Run feature integration and parity tests");
    feature_tests_step.dependOn(&run_integration.step);
    feature_tests_step.dependOn(&run_parity.step);

    const mcp_tests = addTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/integration/mcp_test.zig",
        ctx.build_options_module,
        ctx.abi_module,
    );
    linking.linkIfDarwin(mcp_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const mcp_tests_step = ctx.b.step("mcp-tests", "Run MCP integration tests");
    mcp_tests_step.dependOn(&ctx.b.addRunArtifact(mcp_tests).step);

    // Feature-specific test lanes: unit tests (src/) + integration tests (test/integration/)
    for (feature_lanes) |lane| {
        addFeatureTestLane(ctx, lane.name, lane.display, lane.integration_name orelse lane.name);
    }

    const fmt_paths = &.{ "build.zig", "build", "src", "test" };
    const check_step = ctx.b.step("check", "Run lint + test + parity");
    check_step.dependOn(&ctx.b.addFmt(.{ .paths = fmt_paths, .check = true }).step);
    check_step.dependOn(&run_lib_tests.step);
    check_step.dependOn(&run_parity.step);

    ctx.b.step("cli-tests", "Run CLI tests").dependOn(test_step);
    ctx.b.step("dashboard-smoke", "Run dashboard smoke tests").dependOn(test_step);
    ctx.b.step("validate-flags", "Validate feature flags").dependOn(test_step);
    ctx.b.step("full-check", "Run full check").dependOn(check_step);
    ctx.b.step("verify-all", "Verify all components").dependOn(check_step);

    return .{
        .test_step = test_step,
        .check_parity_step = check_parity_step,
        .feature_tests_step = feature_tests_step,
        .mcp_tests_step = mcp_tests_step,
        .check_step = check_step,
    };
}

/// Create a test compile artifact. When abi_module is non-null, adds it as an "abi" import
/// (for integration tests that use @import("abi")).
fn addTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_path: []const u8,
    build_options_module: *std.Build.Module,
    abi_module: ?*std.Build.Module,
) *std.Build.Step.Compile {
    const test_mod = b.createModule(.{
        .root_source_file = b.path(root_path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addImport("build_options", build_options_module);
    if (abi_module) |m| {
        test_mod.addImport("abi", m);
    }
    return b.addTest(.{ .root_module = test_mod });
}

/// Run the full test suite with a single feature flag overridden.
/// Creates a separate build_options module with the flag toggled,
/// then runs both unit and integration tests under that configuration.
fn addFlagOverrideTestLane(
    ctx: Context,
    comptime flag_name: []const u8,
    flag_value: bool,
    step_name: []const u8,
    step_desc: []const u8,
) void {
    var override_flags = ctx.flags;
    @field(override_flags, flag_name) = flag_value;

    const override_opts = ctx.b.addOptions();
    build_flags.addAllBuildOptions(override_opts, override_flags, ctx.package_version, builtin.zig_version_string);
    const override_bom = override_opts.createModule();

    const module_name = std.fmt.allocPrint(ctx.b.allocator, "abi_{s}", .{step_name}) catch @panic("OOM");
    const override_abi = ctx.b.addModule(module_name, .{
        .root_source_file = ctx.b.path("src/root.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
    });
    override_abi.addImport("build_options", override_bom);

    const lib_tests = ctx.b.addTest(.{
        .root_module = ctx.b.createModule(.{
            .root_source_file = ctx.b.path("src/root.zig"),
            .target = ctx.target,
            .optimize = ctx.optimize,
        }),
    });
    lib_tests.root_module.addImport("build_options", override_bom);
    linking.linkIfDarwin(lib_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);

    const integration_tests = addTests(ctx.b, ctx.target, ctx.optimize, "test/mod.zig", override_bom, override_abi);
    linking.linkIfDarwin(integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);

    const step = ctx.b.step(step_name, step_desc);
    step.dependOn(&ctx.b.addRunArtifact(lib_tests).step);
    step.dependOn(&ctx.b.addRunArtifact(integration_tests).step);
}

/// Add a paired unit + integration test step for a feature.
/// Expects "src/{name}_mod_test.zig" and "test/integration/{integration_name}_test.zig" to exist.
/// Creates a build step named "{display_name}-tests".
fn addFeatureTestLane(ctx: Context, name: []const u8, display_name: []const u8, integration_name: []const u8) void {
    const unit_path = std.fmt.allocPrint(ctx.b.allocator, "src/{s}_mod_test.zig", .{name}) catch @panic("OOM");
    const integration_path = std.fmt.allocPrint(ctx.b.allocator, "test/integration/{s}_test.zig", .{integration_name}) catch @panic("OOM");
    const step_name = std.fmt.allocPrint(ctx.b.allocator, "{s}-tests", .{display_name}) catch @panic("OOM");
    const step_desc = std.fmt.allocPrint(ctx.b.allocator, "Run {s}-focused unit and integration tests", .{display_name}) catch @panic("OOM");

    const unit_tests = addTests(ctx.b, ctx.target, ctx.optimize, unit_path, ctx.build_options_module, null);
    linking.linkIfDarwin(unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const integration_tests = addTests(ctx.b, ctx.target, ctx.optimize, integration_path, ctx.build_options_module, ctx.abi_module);
    linking.linkIfDarwin(integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const step = ctx.b.step(step_name, step_desc);
    step.dependOn(&ctx.b.addRunArtifact(unit_tests).step);
    step.dependOn(&ctx.b.addRunArtifact(integration_tests).step);
}

fn addParityTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options_module: *std.Build.Module,
) *std.Build.Step.Compile {
    const parity_mod = b.createModule(.{
        .root_source_file = b.path("src/feature_parity_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    parity_mod.addImport("build_options", build_options_module);
    return b.addTest(.{ .root_module = parity_mod });
}
