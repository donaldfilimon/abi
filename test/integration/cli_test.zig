//! Integration Tests: CLI
//!
//! Tests CLI data paths through the abi public API.
//! Validates that the data sources used by CLI commands
//! (version, doctor, info, chat) are accessible and functional.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");
const cli = abi.cli;

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

// === Chat Command Path ===

test "cli: persona router routes messages" {
    const persona = abi.ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("What is machine learning?");
    // Should produce a valid routing decision
    try std.testing.expect(decision.confidence >= 0.0);
    try std.testing.expect(decision.confidence <= 1.0);
    try std.testing.expect(decision.reason.len > 0);
}

test "cli: routing decision has valid weights" {
    const persona = abi.ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
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

// === App Builder Path ===

test "cli: app version returns non-empty string" {
    const version = abi.app.version();
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
    _ = builder.withPlugins(abi.config.PluginConfig.withPaths(&.{ "/tmp/abi-plugin.so" }));

    var fw = try builder.build();
    defer fw.deinit();
}

test "cli: untrusted plugin paths are rejected" {
    var builder = abi.App.builder(std.testing.allocator);
    _ = builder.withPlugins(.{
        .paths = &.{"/tmp/abi-untrusted-plugin.so"},
        .allow_untrusted = false,
    });
    try std.testing.expectError(error.InvalidConfig, builder.build());
}

// === Features Command Path (printFeatures data) ===

test "cli: feature catalog has 30 entries" {
    const catalog = abi.meta.features;
    try std.testing.expectEqual(@as(usize, 30), catalog.feature_count);
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
    try std.testing.expect(@hasDecl(@TypeOf(db), "cli"));
}

test "cli: database cli has run function" {
    const db_cli = abi.database.cli;
    try std.testing.expect(@hasDecl(@TypeOf(db_cli), "run"));
}

test "cli: database cli wantsHelp recognizes help flag" {
    const db_cli = abi.database.cli;
    if (@hasDecl(@TypeOf(db_cli), "wantsHelp")) {
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
    try std.testing.expect(@hasDecl(@TypeOf(abi.tui), "dashboard"));
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

test {
    std.testing.refAllDecls(@This());
}
