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
    gateway_tests_step: *std.Build.Step,
    inference_tests_step: *std.Build.Step,
    check_step: *std.Build.Step,
};

pub fn addSteps(ctx: Context) Steps {
    const test_step = ctx.b.step("test", "Run tests");

    const lib_tests = addModuleTests(ctx.b, ctx.target, ctx.optimize, "src/root.zig", ctx.build_options_module);
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(lib_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
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
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const run_integration = ctx.b.addRunArtifact(integration_tests);
    test_step.dependOn(&run_integration.step);

    addTuiTests(ctx);

    const parity_tests = addParityTests(ctx.b, ctx.target, ctx.optimize, ctx.build_options_module);
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(parity_tests, .parity_test, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
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
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(mcp_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const mcp_tests_step = ctx.b.step("mcp-tests", "Run MCP integration tests");
    mcp_tests_step.dependOn(&ctx.b.addRunArtifact(mcp_tests).step);

    const messaging_unit_tests = addModuleTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "src/messaging_mod_test.zig",
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(messaging_unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const messaging_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/messaging_mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(messaging_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const messaging_tests_step = ctx.b.step("messaging-tests", "Run messaging-focused unit and integration tests");
    messaging_tests_step.dependOn(&ctx.b.addRunArtifact(messaging_unit_tests).step);
    messaging_tests_step.dependOn(&ctx.b.addRunArtifact(messaging_integration_tests).step);

    const secrets_unit_tests = addModuleTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "src/secrets_mod_test.zig",
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(secrets_unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const secrets_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/secrets_mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(secrets_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const secrets_tests_step = ctx.b.step("secrets-tests", "Run secrets-focused unit and integration tests");
    secrets_tests_step.dependOn(&ctx.b.addRunArtifact(secrets_unit_tests).step);
    secrets_tests_step.dependOn(&ctx.b.addRunArtifact(secrets_integration_tests).step);

    const pitr_unit_tests = addModuleTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "src/pitr_mod_test.zig",
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(pitr_unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const pitr_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/pitr_mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(pitr_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const pitr_tests_step = ctx.b.step("pitr-tests", "Run PITR-focused unit and integration tests");
    pitr_tests_step.dependOn(&ctx.b.addRunArtifact(pitr_unit_tests).step);
    pitr_tests_step.dependOn(&ctx.b.addRunArtifact(pitr_integration_tests).step);

    const agents_unit_tests = addModuleTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "src/agents_mod_test.zig",
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(agents_unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const agents_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/agents_mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(agents_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const agents_tests_step = ctx.b.step("agents-tests", "Run agents-focused unit and integration tests");
    agents_tests_step.dependOn(&ctx.b.addRunArtifact(agents_unit_tests).step);
    agents_tests_step.dependOn(&ctx.b.addRunArtifact(agents_integration_tests).step);

    const gateway_unit_tests = addModuleTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "src/gateway_mod_test.zig",
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(gateway_unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const gateway_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/gateway_mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(gateway_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const gateway_tests_step = ctx.b.step("gateway-tests", "Run gateway-focused unit and integration tests");
    gateway_tests_step.dependOn(&ctx.b.addRunArtifact(gateway_unit_tests).step);
    gateway_tests_step.dependOn(&ctx.b.addRunArtifact(gateway_integration_tests).step);

    const inference_unit_tests = addModuleTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "src/inference_mod_test.zig",
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(inference_unit_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const inference_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/inference_mod.zig",
        ctx.abi_module,
        ctx.build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(inference_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }
    const inference_tests_step = ctx.b.step("inference-tests", "Run inference-focused unit and integration tests");
    inference_tests_step.dependOn(&ctx.b.addRunArtifact(inference_unit_tests).step);
    inference_tests_step.dependOn(&ctx.b.addRunArtifact(inference_integration_tests).step);

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
        .gateway_tests_step = gateway_tests_step,
        .inference_tests_step = inference_tests_step,
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
    return b.addTest(.{ .root_module = test_mod });
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
    return b.addTest(.{ .root_module = integration_mod });
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
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(tui_lib_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }

    const tui_integration_tests = addIntegrationTests(
        ctx.b,
        ctx.target,
        ctx.optimize,
        "test/mod.zig",
        tui_abi_module,
        tui_build_options_module,
    );
    if (ctx.target.result.os.tag == .macos) {
        linking.linkDarwinArtifact(tui_integration_tests, .test_artifact, ctx.flags.feat_gpu, ctx.flags.gpu_metal);
    }

    const tui_tests_step = ctx.b.step("tui-tests", "Run TUI tests with feat-tui=true");
    tui_tests_step.dependOn(&ctx.b.addRunArtifact(tui_lib_tests).step);
    tui_tests_step.dependOn(&ctx.b.addRunArtifact(tui_integration_tests).step);
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
