const std = @import("std");
const abi = @import("abi");

test "vnext capability roundtrip with core feature enum" {
    const cap = abi.vnext.capability.fromFeature(abi.Feature.gpu);
    try std.testing.expectEqualStrings("gpu", cap.name());

    const feature = abi.vnext.capability.toFeature(abi.Feature, cap);
    try std.testing.expectEqual(abi.Feature.gpu, feature);
}

test "vnext app init supports minimal configuration" {
    var app = try abi.vnext.App.init(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    defer app.deinit();

    try std.testing.expectEqual(abi.Framework.State.running, app.getFrameworkConst().getState());
}

test "vnext app strict capability check fails when capability is missing" {
    const result = abi.vnext.App.init(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
        .strict_capability_check = true,
        .required_capabilities = &.{abi.vnext.Capability.gpu},
    });
    try std.testing.expectError(error.CapabilityUnavailable, result);
}

test "legacy abi init/initDefault and vnext App init paths are behaviorally equivalent" {
    const cfg = abi.Config.defaults();
    var legacy = try abi.init(std.testing.allocator, cfg);
    defer legacy.deinit();

    const app_cfg = abi.vnext.AppConfig{
        .framework = cfg,
    };
    var app = try abi.vnext.App.init(std.testing.allocator, app_cfg);
    defer app.deinit();

    try std.testing.expectEqual(legacy.getState(), app.getFrameworkConst().getState());

    inline for (std.meta.fields(abi.Feature)) |field| {
        const feature: abi.Feature = @enumFromInt(field.value);
        try std.testing.expectEqual(
            legacy.isEnabled(feature),
            app.getFrameworkConst().isEnabled(feature),
        );
    }
}

test "legacy initDefault maps to vnext initDefault" {
    var legacy = try abi.initDefault(std.testing.allocator);
    defer legacy.deinit();

    var app = try abi.vnext.App.initDefault(std.testing.allocator);
    defer app.deinit();

    try std.testing.expectEqual(legacy.getState(), app.getFrameworkConst().getState());
}

test "abi.initApp and abi.vnext.App.init are equivalent wrappers" {
    const cfg = abi.vnext.AppConfig{
        .framework = abi.Config.minimal(),
        .strict_capability_check = false,
    };

    var app_from_wrapper = try abi.initApp(std.testing.allocator, cfg);
    defer app_from_wrapper.deinit();

    var app_direct = try abi.vnext.App.init(std.testing.allocator, cfg);
    defer app_direct.deinit();

    try std.testing.expectEqual(
        app_from_wrapper.getFrameworkConst().getState(),
        app_direct.getFrameworkConst().getState(),
    );
}

test "vnext capability mapping roundtrips every feature variant" {
    inline for (std.meta.fields(abi.Feature)) |field| {
        const feature: abi.Feature = @enumFromInt(field.value);
        const cap = abi.vnext.capability.fromFeature(feature);
        const back = abi.vnext.capability.toFeature(abi.Feature, cap);
        try std.testing.expectEqual(feature, back);
    }
}

// ============================================================================
// vNext new API tests
// ============================================================================

test "vnext App.start is equivalent to App.init" {
    var app_start = try abi.vnext.App.start(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    defer app_start.stop();

    var app_init = try abi.vnext.App.init(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    defer app_init.deinit();

    try std.testing.expectEqual(app_start.state(), app_init.getFrameworkConst().getState());
}

test "vnext app.state() returns running after start" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{});
    defer app.stop();

    try std.testing.expectEqual(abi.vnext.App.State.running, app.state());
}

test "vnext app.has() matches framework.isEnabled() for all features" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{});
    defer app.stop();

    inline for (std.meta.fields(abi.Feature)) |field| {
        const feat: abi.Feature = @enumFromInt(field.value);
        const cap = abi.vnext.capability.fromFeature(feat);
        try std.testing.expectEqual(
            app.getFrameworkConst().isEnabled(feat),
            app.has(cap),
        );
    }
}

test "vnext app.feature(.gpu) returns null for minimal config" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    defer app.stop();

    try std.testing.expect(app.feature(.gpu) == null);
    try std.testing.expect(app.feature(.database) == null);
    try std.testing.expect(app.feature(.ai) == null);
}

test "vnext app.runtime() returns non-null" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    defer app.stop();

    const rt = app.runtime();
    try std.testing.expect(@intFromPtr(rt) != 0);
}

test "vnext app.registry() compiles and returns" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    defer app.stop();

    const reg = app.registry();
    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

test "vnext app.stop() performs clean shutdown" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    app.stop();
    // If we reach here without crash/leak, stop() worked.
}

test "vnext app.stopWithOptions with timeout returns true" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    const ok = app.stopWithOptions(.{ .timeout_ms = 5000 });
    try std.testing.expect(ok);
}

test "vnext app.stopWithOptions without timeout returns true" {
    var app = try abi.vnext.App.start(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    const ok = app.stopWithOptions(.{});
    try std.testing.expect(ok);
}

test "vnext abi.startApp is equivalent to App.start" {
    var app = try abi.startApp(std.testing.allocator, .{
        .framework = abi.Config.minimal(),
    });
    defer app.stop();

    try std.testing.expectEqual(abi.vnext.App.State.running, app.state());
}

test "vnext AppBuilder creates working App" {
    var builder = abi.vnext.App.configure(std.testing.allocator);
    var app = try builder.start();
    defer app.stop();

    try std.testing.expectEqual(abi.vnext.App.State.running, app.state());
}
