//! Integration Tests: CLI
//!
//! Tests CLI data paths through the abi public API.
//! Validates that the data sources used by CLI commands
//! (version, doctor, info, chat) are accessible and functional.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");
const cli = abi.cli;

fn renderStatus() ![]u8 {
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer writer.deinit();
    try cli.writeStatus(&writer.writer);
    return writer.toOwnedSlice();
}

fn renderHelp() ![]u8 {
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer writer.deinit();
    try cli.writeHelp(&writer.writer);
    return writer.toOwnedSlice();
}

fn renderServeHelp() ![]u8 {
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer writer.deinit();
    try cli.writeServeHelp(&writer.writer);
    return writer.toOwnedSlice();
}

fn renderChatReport(options: cli.RenderOptions) ![]u8 {
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer writer.deinit();
    try cli.writeChatPipelineReport(&writer.writer, options, .{
        .input = "hello world",
        .primary = "Abbey",
        .strategy = "blend",
        .reason = "Test route",
        .confidence_pct = 82.0,
        .abbey_pct = 70.0,
        .aviva_pct = 20.0,
        .abi_pct = 10.0,
    });
    return writer.toOwnedSlice();
}
// === Version Command Path ===

test "cli: package version is non-empty" {
    const version = abi.meta.package_version;
    try std.testing.expect(version.len > 0);
}

test "cli: version function returns package version" {
    const v1 = abi.version();
    const v2 = abi.meta.version();
    try std.testing.expectEqualStrings(v1, v2);
}

// === Doctor Command Path ===

test "cli: build_options feature flags are accessible" {
    // Verify all feature flags the doctor command prints are readable
    const flags = .{
        build_options.feat_ai,
        build_options.feat_gpu,
        build_options.feat_database,
        build_options.feat_network,
        build_options.feat_web,
        build_options.feat_search,
        build_options.feat_cache,
        build_options.feat_auth,
        build_options.feat_lsp,
        build_options.feat_mcp,
        build_options.feat_mobile,
        build_options.feat_desktop,
    };
    // Each flag is a bool — just verify they don't crash when accessed
    inline for (flags) |flag| {
        try std.testing.expect(flag == true or flag == false);
    }
}

test "cli: AI sub-feature flags are accessible" {
    const ai_flags = .{
        build_options.feat_llm,
        build_options.feat_training,
        build_options.feat_vision,
        build_options.feat_reasoning,
    };
    inline for (ai_flags) |flag| {
        try std.testing.expect(flag == true or flag == false);
    }
}

test "cli: GPU backend flags are accessible" {
    const gpu_flags = .{
        build_options.gpu_metal,
        build_options.gpu_cuda,
        build_options.gpu_vulkan,
        build_options.gpu_stdgpu,
    };
    inline for (gpu_flags) |flag| {
        try std.testing.expect(flag == true or flag == false);
    }
}

// === Info Command Path ===

test "cli: feature catalog has entries" {
    const catalog = abi.meta.features;
    try std.testing.expect(catalog.all.len > 0);
    try std.testing.expect(catalog.feature_count > 20);
}

test "cli: feature catalog contains expected features" {
    const catalog = abi.meta.features;
    // Verify key features exist in catalog via the description free function
    try std.testing.expectEqualStrings("GPU acceleration and compute", catalog.description(.gpu));
    try std.testing.expectEqualStrings("AI core functionality", catalog.description(.ai));
    try std.testing.expectEqualStrings("Vector database (WDBX)", catalog.description(.database));
}

test "cli: feature catalog metadata is consistent" {
    const catalog = abi.meta.features;
    for (catalog.all, 0..) |entry, idx| {
        // Each feature's enum ordinal matches its position in the array
        try std.testing.expectEqual(entry.feature, @as(catalog.Feature, @enumFromInt(idx)));
        // Description is non-empty
        try std.testing.expect(entry.description.len > 0);
        // Compile flag field is non-empty
        try std.testing.expect(entry.compile_flag_field.len > 0);
        // Module paths are non-empty
        try std.testing.expect(entry.real_module_path.len > 0);
        try std.testing.expect(entry.stub_module_path.len > 0);
    }
}

test "cli: single-token commands are described in parity with runtime" {
    const expected_commands = [_][]const u8{
        "version",
        "doctor",
        "features",
        "platform",
        "connectors",
        "info",
        "serve",
        "dashboard",
        "lsp",
    };

    try std.testing.expectEqual(expected_commands.len, cli.single_token_commands.len);

    for (expected_commands) |expected| {
        var found = false;
        for (cli.single_token_commands) |descriptor| {
            if (std.mem.eql(u8, descriptor.name, expected)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }

    for (cli.single_token_commands) |descriptor| {
        var found_displayed = false;
        for (cli.displayed_commands) |displayed| {
            if (std.mem.eql(u8, displayed.usage, descriptor.name)) {
                found_displayed = true;
                break;
            }
        }
        try std.testing.expect(found_displayed);
    }
}

test "cli: help output renders the displayed command catalog" {
    const help = try renderHelp();
    defer std.testing.allocator.free(help);

    try std.testing.expect(std.mem.indexOf(u8, help, "Diagnostics:") != null);
    try std.testing.expect(std.mem.indexOf(u8, help, "AI & Data:") != null);
    try std.testing.expect(std.mem.indexOf(u8, help, "Interactive:") != null);
    try std.testing.expect(std.mem.indexOf(u8, help, "Build:") != null);

    for (cli.displayed_commands) |command| {
        try std.testing.expect(std.mem.indexOf(u8, help, command.usage) != null);
    }

    try std.testing.expect(std.mem.indexOf(u8, help, cli.dashboard_command_detail) != null);
    try std.testing.expect(std.mem.indexOf(u8, help, cli.dashboard_fallback_note) != null);
}

test "cli: status output includes shared feature-gated command tags" {
    const status = try renderStatus();
    defer std.testing.allocator.free(status);

    try std.testing.expect(std.mem.indexOf(u8, status, abi.meta.package_version) != null);
    for (cli.displayed_commands) |command| {
        try std.testing.expect(std.mem.indexOf(u8, status, command.usage) != null);
    }
    try std.testing.expect(std.mem.indexOf(u8, status, cli.dashboard_command_detail) != null);
    try std.testing.expect(std.mem.indexOf(u8, status, cli.dashboard_fallback_note) != null);
    try std.testing.expect(std.mem.indexOf(u8, status, if (build_options.feat_database) "[enabled]" else "[disabled]") != null);
    try std.testing.expect(std.mem.indexOf(u8, status, if (build_options.feat_tui) "[enabled]" else "[disabled]") != null);
}

test "cli: help and status share dashboard contract wording" {
    const help = try renderHelp();
    defer std.testing.allocator.free(help);
    const status = try renderStatus();
    defer std.testing.allocator.free(status);

    try std.testing.expect(std.mem.indexOf(u8, help, cli.dashboard_command_detail) != null);
    try std.testing.expect(std.mem.indexOf(u8, status, cli.dashboard_command_detail) != null);
    try std.testing.expect(std.mem.indexOf(u8, help, cli.dashboard_fallback_note) != null);
    try std.testing.expect(std.mem.indexOf(u8, status, cli.dashboard_fallback_note) != null);
}

test "cli: serve help uses the shared writer path" {
    const help = try renderServeHelp();
    defer std.testing.allocator.free(help);

    try std.testing.expect(std.mem.indexOf(u8, help, "Usage: abi serve [options]") != null);
    try std.testing.expect(std.mem.indexOf(u8, help, "--addr <host:port>") != null);
}

test "cli: chat pipeline report filters the header in pipeline mode" {
    const report = try renderChatReport(.{ .stdout_is_tty = false });
    defer std.testing.allocator.free(report);

    try std.testing.expect(std.mem.indexOf(u8, report, "ABI Chat - Profile Pipeline") == null);
    try std.testing.expect(std.mem.indexOf(u8, report, "Input: hello world") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "Execution:") != null);
}

test "cli: chat pipeline report keeps the header for tty output" {
    const report = try renderChatReport(.{ .stdout_is_tty = true });
    defer std.testing.allocator.free(report);

    try std.testing.expect(std.mem.indexOf(u8, report, "ABI Chat - Profile Pipeline") != null);
}

// === Chat Command Path ===

test "cli: profile router routes messages" {
    const profile = abi.ai.profile;
    var registry = profile.ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = profile.MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("What is machine learning?");
    // Should produce a valid routing decision
    try std.testing.expect(decision.confidence >= 0.0);
    try std.testing.expect(decision.confidence <= 1.0);
    try std.testing.expect(decision.reason.len > 0);
}

test "cli: routing decision has valid weights" {
    const profile = abi.ai.profile;
    var registry = profile.ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = profile.MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("Help me debug this code");
    // Each weight should be in valid range
    try std.testing.expect(decision.weights.abbey >= 0.0);
    try std.testing.expect(decision.weights.abbey <= 1.0);
    try std.testing.expect(decision.weights.aviva >= 0.0);
    try std.testing.expect(decision.weights.aviva <= 1.0);
    try std.testing.expect(decision.weights.abi >= 0.0);
    try std.testing.expect(decision.weights.abi <= 1.0);
}

test "cli: chat message helper joins tokenized args" {
    const message_args = [_][:0]const u8{ "Hello,", "how", "are", "you?" };
    const message = try cli.joinChatMessage(std.testing.allocator, &message_args);
    defer std.testing.allocator.free(message);

    try std.testing.expectEqualStrings("Hello, how are you?", message);
}

test "cli: chat message helper handles single arg" {
    const single = [_][:0]const u8{"hello"};
    const message = try cli.joinChatMessage(std.testing.allocator, &single);
    defer std.testing.allocator.free(message);

    try std.testing.expectEqualStrings("hello", message);
}

test "cli: chat message helper handles empty args" {
    const empty: []const [:0]const u8 = &.{};
    const message = try cli.joinChatMessage(std.testing.allocator, empty);
    defer std.testing.allocator.free(message);

    try std.testing.expectEqualStrings("", message);
}

// === App Builder Path ===

test "cli: app version returns non-empty string" {
    const version = abi.version();
    try std.testing.expect(version.len > 0);
}

// === Status Command Path (printStatus data) ===

test "cli: status enabled count matches catalog iteration" {
    // Replicate the comptime enabled-feature count used by printStatus()
    const catalog = abi.meta.features;
    const enabled = comptime blk: {
        var count: u32 = 0;
        for (catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) count += 1;
        }
        break :blk count;
    };
    // At least some features should be enabled in the default build
    try std.testing.expect(enabled > 0);
    try std.testing.expect(enabled <= catalog.feature_count);
}

test "cli: status uses package_version from build_options" {
    // printStatus reads build_options.package_version — verify it's consistent
    const bo_version = build_options.package_version;
    const meta_version = abi.meta.package_version;
    try std.testing.expectEqualStrings(bo_version, meta_version);
}

test "cli: serve routing recognizes acp alias" {
    try std.testing.expect(cli.isServeInvocation(&.{"serve"}));
    try std.testing.expect(cli.isServeInvocation(&.{ "acp", "serve" }));
    try std.testing.expect(!cli.isServeInvocation(&.{ "acp", "status" }));
}

test "cli: serve address parsing honors addr and port flags" {
    const port_args = [_][:0]const u8{ "--port", "9090" };
    const port_address = try cli.parseServeAddress(std.testing.allocator, &port_args);
    defer std.testing.allocator.free(port_address);
    try std.testing.expectEqualStrings("127.0.0.1:9090", port_address);

    const ipv6_args = [_][:0]const u8{ "--host", "::1", "--port", "9090" };
    const ipv6_address = try cli.parseServeAddress(std.testing.allocator, &ipv6_args);
    defer std.testing.allocator.free(ipv6_address);
    try std.testing.expectEqualStrings("[::1]:9090", ipv6_address);

    const addr_args = [_][:0]const u8{ "--addr", "0.0.0.0:8080" };
    const explicit_address = try cli.parseServeAddress(std.testing.allocator, &addr_args);
    defer std.testing.allocator.free(explicit_address);
    try std.testing.expectEqualStrings("0.0.0.0:8080", explicit_address);
}

test "cli: plugin path helper builds successfully" {
    var builder = abi.App.builder(std.testing.allocator);
    _ = builder.withPlugins(abi.config.PluginConfig.withPaths(&.{"/tmp/abi-plugin.so"}));

    var fw = try builder.build();
    defer fw.deinit();
}
// === Features Command Path (printFeatures data) ===

test "cli: feature catalog count matches exported entries" {
    const catalog = abi.meta.features;
    try std.testing.expectEqual(catalog.all.len, catalog.feature_count);
    try std.testing.expectEqual(@typeInfo(catalog.Feature).@"enum".fields.len, catalog.feature_count);
}

test "cli: every feature has a valid compile flag in build_options" {
    const catalog = abi.meta.features;
    // printFeatures() reads @field(build_options, entry.compile_flag_field) for each.
    // Verify all compile flags produce a bool (true or false) at comptime.
    inline for (catalog.all) |entry| {
        const enabled: bool = @field(build_options, entry.compile_flag_field);
        try std.testing.expect(enabled == true or enabled == false);
    }
}

test "cli: feature names are unique and non-empty" {
    const catalog = abi.meta.features;
    for (catalog.all, 0..) |a, i| {
        try std.testing.expect(a.feature.name().len > 0);
        // Verify uniqueness: no two entries share the same feature enum
        for (catalog.all[0..i]) |b| {
            try std.testing.expect(a.feature != b.feature);
        }
    }
}

test "cli: parent features reference valid catalog entries" {
    const catalog = abi.meta.features;
    for (catalog.all) |entry| {
        if (entry.parent) |parent_feat| {
            // The parent must also exist in the catalog
            var found = false;
            for (catalog.all) |other| {
                if (other.feature == parent_feat) {
                    found = true;
                    break;
                }
            }
            try std.testing.expect(found);
        }
    }
}

// === Platform Command Path (printPlatform data) ===

test "cli: platform getPlatformInfo returns valid OS and arch" {
    const platform = abi.platform;
    const info = platform.getPlatformInfo();
    // OS tag name should be non-empty
    try std.testing.expect(@tagName(info.os).len > 0);
    // Arch tag name should be non-empty
    try std.testing.expect(@tagName(info.arch).len > 0);
}

test "cli: platform getDescription returns non-empty string" {
    const desc = abi.platform.getDescription();
    try std.testing.expect(desc.len > 0);
}

test "cli: platform getCpuCount returns at least 1" {
    const cpus = abi.platform.getCpuCount();
    try std.testing.expect(cpus >= 1);
}

test "cli: platform supportsThreading returns a bool" {
    const val = abi.platform.supportsThreading();
    try std.testing.expect(val == true or val == false);
}

// === Database CLI Command Path (runDb data) ===

test "cli: database module exposes cli decl" {
    // runDb() checks @hasDecl(db, "cli") — verify it's present
    const db = abi.database;
    try std.testing.expect(@hasDecl(db, "cli"));
}

test "cli: database cli has run function" {
    const db_cli = abi.database.cli;
    try std.testing.expect(@hasDecl(db_cli, "run"));
}

test "cli: database diagnostics type is accessible and healthy by default" {
    const DiagnosticsInfo = abi.database.DiagnosticsInfo;
    const info: DiagnosticsInfo = .{
        .name = "test",
        .vector_count = 0,
        .dimension = 0,
        .memory = undefined,
        .config = undefined,
        .pool_stats = null,
        .index_health = 1.0,
        .norm_cache_health = 1.0,
    };
    try std.testing.expect(info.isHealthy());
    try std.testing.expectEqual(@as(f32, 1.0), info.index_health);
    try std.testing.expectEqual(@as(f32, 1.0), info.norm_cache_health);
    try std.testing.expect(info.pool_stats == null);
}

test "cli: database cli wantsHelp recognizes help flag" {
    const db_cli = abi.database.cli;
    if (@hasDecl(db_cli, "wantsHelp")) {
        const help_args = [_][:0]const u8{"--help"};
        try std.testing.expect(db_cli.wantsHelp(&help_args));

        const no_help_args = [_][:0]const u8{"stats"};
        try std.testing.expect(!db_cli.wantsHelp(&no_help_args));
    }
}

// === Dashboard Command Path (runDashboard data) ===

test "cli: tui module is accessible" {
    // runDashboard() accesses root.tui — verify the module exists
    const tui = abi.tui;
    _ = tui;
}

test "cli: tui has dashboard decl" {
    // runDashboard() accesses root.tui.dashboard
    try std.testing.expect(@hasDecl(abi.tui, "dashboard"));
}

test "cli: tui isEnabled reflects build option" {
    const enabled = abi.tui.isEnabled();
    try std.testing.expectEqual(build_options.feat_tui, enabled);
}

// === Connectors Command Path (printConnectors data) ===

test "cli: connectors module is accessible" {
    // printConnectors() prints static text, but the module should be reachable
    const conn = abi.connectors;
    _ = conn;
}

// === Plugin System ===

var mock_plugin_initialized = false;

fn mock_init_plugin(ptr: ?*anyopaque, fw: *anyopaque) anyerror!void {
    _ = ptr;
    _ = fw;
    mock_plugin_initialized = true;
}

test "cli: plugin system can inject and initialize mock plugin" {
    mock_plugin_initialized = false;
    var builder = abi.App.builder(std.testing.allocator);
    _ = builder.withDefaults();

    // Register the static plugin
    _ = builder.registerStaticPlugin(null, mock_init_plugin);

    var fw = try builder.build();
    defer fw.deinit();

    try std.testing.expect(mock_plugin_initialized);
}

test "cli: untrusted plugin paths are rejected" {
    var builder = abi.App.builder(std.testing.allocator);
    _ = builder.withPlugins(.{
        .paths = &.{"/tmp/abi-untrusted-plugin.so"},
        .allow_untrusted = false,
    });

    try std.testing.expectError(error.InvalidConfig, builder.build());
}

test {
    std.testing.refAllDecls(@This());
}
