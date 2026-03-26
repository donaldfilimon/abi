const std = @import("std");
const build_flags = @import("flags.zig");
const linking = @import("linking.zig");

pub const Context = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    flags: build_flags.FeatureFlags,
    build_options_module: *std.Build.Module,
    abi_module: *std.Build.Module,
};

pub const Steps = struct {
    test_step: *std.Build.Step,
    check_parity_step: *std.Build.Step,
    feature_tests_step: *std.Build.Step,
    mcp_tests_step: *std.Build.Step,
    messaging_tests_step: *std.Build.Step,
    secrets_tests_step: *std.Build.Step,
    pitr_tests_step: *std.Build.Step,
    agents_tests_step: *std.Build.Step,
    multi_agent_tests_step: *std.Build.Step,
    orchestration_tests_step: *std.Build.Step,
    gateway_tests_step: *std.Build.Step,
    inference_tests_step: *std.Build.Step,
    gpu_tests_step: *std.Build.Step,
    network_tests_step: *std.Build.Step,
    web_tests_step: *std.Build.Step,
    observability_tests_step: *std.Build.Step,
    search_tests_step: *std.Build.Step,
    auth_tests_step: *std.Build.Step,
    storage_tests_step: *std.Build.Step,
    cloud_tests_step: *std.Build.Step,
    cache_tests_step: *std.Build.Step,
    database_tests_step: *std.Build.Step,
    connectors_tests_step: *std.Build.Step,
    lsp_tests_step: *std.Build.Step,
    acp_tests_step: *std.Build.Step,
    ha_tests_step: *std.Build.Step,
    tasks_tests_step: *std.Build.Step,
    documents_tests_step: *std.Build.Step,
    compute_tests_step: *std.Build.Step,
    desktop_tests_step: *std.Build.Step,
    pipeline_tests_step: *std.Build.Step,
    check_step: *std.Build.Step,
};

pub fn addSteps(ctx: Context) Steps {
    const test_step = ctx.b.step("test", "Run tests");

    const lib_tests = addModuleTests(ctx.b, ctx.target, ctx.optimize, "src/root.zig", ctx.build_options_module);
    linking.linkIfDarwin(lib_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const run_lib_tests = ctx.b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_lib_tests.step);

    const integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    linking.linkIfDarwin(integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const run_integration = ctx.b.addRunArtifact(integration_tests);
    test_step.dependOn(&run_integration.step);

    addTuiTests(ctx);

    const parity_tests = addParityTests(ctx.b, ctx.target, ctx.optimize, ctx.build_options_module);
    linking.linkIfDarwin(parity_tests, .parity_test, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const run_parity = ctx.b.addRunArtifact(parity_tests);
    const check_parity_step = ctx.b.step("check-parity", "Verify mod/stub declaration parity");
    check_parity_step.dependOn(&run_parity.step);

    const feature_tests_step = ctx.b.step("feature-tests", "Run feature integration and parity tests");
    feature_tests_step.dependOn(&run_integration.step);
    feature_tests_step.dependOn(&run_parity.step);

    const mcp_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/mcp_mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    linking.linkIfDarwin(mcp_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const mcp_tests_step = ctx.b.step("mcp-tests", "Run MCP integration tests");
    mcp_tests_step.dependOn(&ctx.b.addRunArtifact(mcp_tests).step);

    // Feature-specific test lanes: unit tests (src/) + integration tests (test/)
    const messaging_tests_step = addFeatureTestLane(ctx, "messaging", "messaging");
    const secrets_tests_step = addFeatureTestLane(ctx, "secrets", "secrets");
    const pitr_tests_step = addFeatureTestLane(ctx, "pitr", "PITR");
    const agents_tests_step = addFeatureTestLane(ctx, "agents", "agents");
    const multi_agent_tests_step = addFeatureTestLane(ctx, "multi_agent", "multi-agent");
    const orchestration_tests_step = addFeatureTestLane(ctx, "orchestration", "orchestration");
    const gateway_tests_step = addFeatureTestLane(ctx, "gateway", "gateway");
    const inference_tests_step = addFeatureTestLane(ctx, "inference", "inference");
    const gpu_tests_step = addFeatureTestLane(ctx, "gpu", "gpu");
    const network_tests_step = addFeatureTestLane(ctx, "network", "network");
    const web_tests_step = addFeatureTestLane(ctx, "web", "web");
    const observability_tests_step = addFeatureTestLane(ctx, "observability", "observability");
    const search_tests_step = addFeatureTestLane(ctx, "search", "search");
    const auth_tests_step = addFeatureTestLane(ctx, "auth", "auth");
    const storage_tests_step = addFeatureTestLane(ctx, "storage", "storage");
    const cloud_tests_step = addFeatureTestLane(ctx, "cloud", "cloud");
    const cache_tests_step = addFeatureTestLane(ctx, "cache", "cache");
    const database_tests_step = addFeatureTestLane(ctx, "database", "database");
    const connectors_tests_step = addFeatureTestLane(ctx, "connectors", "connectors");
    const lsp_tests_step = addFeatureTestLane(ctx, "lsp", "lsp");
    const acp_tests_step = addFeatureTestLane(ctx, "acp", "acp");
    const ha_tests_step = addFeatureTestLane(ctx, "ha", "ha");
    const tasks_tests_step = addFeatureTestLane(ctx, "tasks", "tasks");
    const documents_tests_step = addFeatureTestLane(ctx, "documents", "documents");
    const compute_tests_step = addFeatureTestLane(ctx, "compute", "compute");
    const desktop_tests_step = addFeatureTestLane(ctx, "desktop", "desktop");
    const pipeline_tests_step = addFeatureTestLane(ctx, "pipeline", "pipeline");

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
        .messaging_tests_step = messaging_tests_step,
        .secrets_tests_step = secrets_tests_step,
        .pitr_tests_step = pitr_tests_step,
        .agents_tests_step = agents_tests_step,
        .multi_agent_tests_step = multi_agent_tests_step,
        .orchestration_tests_step = orchestration_tests_step,
        .gateway_tests_step = gateway_tests_step,
        .inference_tests_step = inference_tests_step,
        .gpu_tests_step = gpu_tests_step,
        .network_tests_step = network_tests_step,
        .web_tests_step = web_tests_step,
        .observability_tests_step = observability_tests_step,
        .search_tests_step = search_tests_step,
        .auth_tests_step = auth_tests_step,
        .storage_tests_step = storage_tests_step,
        .cloud_tests_step = cloud_tests_step,
        .cache_tests_step = cache_tests_step,
        .database_tests_step = database_tests_step,
        .connectors_tests_step = connectors_tests_step,
        .lsp_tests_step = lsp_tests_step,
        .acp_tests_step = acp_tests_step,
        .ha_tests_step = ha_tests_step,
        .tasks_tests_step = tasks_tests_step,
        .documents_tests_step = documents_tests_step,
        .compute_tests_step = compute_tests_step,
        .desktop_tests_step = desktop_tests_step,
        .pipeline_tests_step = pipeline_tests_step,
        .check_step = check_step,
    };
}

fn addModuleTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_path: []const u8,
    build_options_module: *std.Build.Module,
) *std.Build.Step.Compile {
    const test_mod = b.createModule(.{
        .root_source_file = b.path(root_path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addImport("build_options", build_options_module);
    const tests = b.addTest(.{ .root_module = test_mod });
    return tests;
}

fn addIntegrationTests(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    root_path: []const u8,
    abi_module: *std.Build.Module,
    build_options_module: *std.Build.Module,
) *std.Build.Step.Compile {
    const integration_mod = b.createModule(.{
        .root_source_file = b.path(root_path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    integration_mod.addImport("abi", abi_module);
    integration_mod.addImport("build_options", build_options_module);
    const tests = b.addTest(.{ .root_module = integration_mod });
    return tests;
}

fn addTuiTests(ctx: Context) void {
    var tui_flags = ctx.flags;
    tui_flags.feat_tui = true;

    const tui_build_opts = ctx.b.addOptions();
    build_flags.addAllBuildOptions(tui_build_opts, tui_flags);
    const tui_build_options_module = tui_build_opts.createModule();

    const tui_abi_module = ctx.b.addModule("abi_tui", .{
        .root_source_file = ctx.b.path("src/root.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
    });
    tui_abi_module.addImport("build_options", tui_build_options_module);

    const tui_lib_tests = ctx.b.addTest(.{
        .root_module = ctx.b.createModule(.{
            .root_source_file = ctx.b.path("src/root.zig"),
            .target = ctx.target,
            .optimize = ctx.optimize,
        }),
    });
    tui_lib_tests.root_module.addImport("build_options", tui_build_options_module);
    linking.linkIfDarwin(tui_lib_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);

    const tui_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/mod.zig",
        tui_abi_module,
        tui_build_options_module,
    );
    linking.linkIfDarwin(tui_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);

    const tui_tests_step = ctx.b.step("tui-tests", "Run TUI tests with feat-tui=true");
    tui_tests_step.dependOn(&ctx.b.addRunArtifact(tui_lib_tests).step);
    tui_tests_step.dependOn(&ctx.b.addRunArtifact(tui_integration_tests).step);
}

/// Add a paired unit + integration test step for a feature.
/// Expects "src/{name}_mod_test.zig" and "test/{name}_mod.zig" to exist.
/// Creates a build step named "{display_name}-tests".
fn addFeatureTestLane(ctx: Context, comptime name: []const u8, comptime display_name: []const u8) *std.Build.Step {
    const unit_tests = addModuleTests(ctx.b, ctx.target, ctx.optimize, "src/" ++ name ++ "_mod_test.zig", ctx.build_options_module);
    linking.linkIfDarwin(unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const integration_tests = addIntegrationTests(ctx.b, ctx.target, ctx.optimize, "test/" ++ name ++ "_mod.zig", ctx.abi_module, ctx.build_options_module);
    linking.linkIfDarwin(integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    const step = ctx.b.step(display_name ++ "-tests", "Run " ++ display_name ++ "-focused unit and integration tests");
    step.dependOn(&ctx.b.addRunArtifact(unit_tests).step);
    step.dependOn(&ctx.b.addRunArtifact(integration_tests).step);
    return step;
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
    const tests = b.addTest(.{ .root_module = parity_mod });
    return tests;
}
