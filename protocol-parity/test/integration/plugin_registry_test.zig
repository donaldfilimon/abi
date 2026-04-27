//! Integration Tests: Plugin Registry
//!
//! Exercises the plugin ecosystem through the `abi.registry.plugin` public
//! surface.  Tests cover registration, lifecycle state transitions, callback
//! invocation, error paths, capability queries, and full register-load-unload
//! cycles.

const std = @import("std");
const abi = @import("abi");

const plugin = abi.registry.plugin;

const PluginRegistry = plugin.PluginRegistry;
const PluginDescriptor = plugin.PluginDescriptor;
const PluginCallbacks = plugin.PluginCallbacks;
const PluginCapability = plugin.PluginCapability;
const PluginState = plugin.PluginState;
const PluginInfo = plugin.PluginInfo;
const PluginError = plugin.PluginError;
const PluginApi = abi.registry.types.PluginApi;

// ── Helpers ────────────────────────────────────────────────────────────

/// Mutable flag set by callback helpers so tests can observe side-effects.
var load_counter: u32 = 0;
var unload_counter: u32 = 0;
var fail_on_load: bool = false;

fn countingOnLoad() anyerror!void {
    if (fail_on_load) return error.SimulatedLoadFailure;
    load_counter += 1;
}

fn countingOnUnload() void {
    unload_counter += 1;
}

fn resetCounters() void {
    load_counter = 0;
    unload_counter = 0;
    fail_on_load = false;
}

/// Build a descriptor with sensible defaults and the given name / capabilities.
fn makeDescriptor(name: []const u8, caps: []const PluginCapability) PluginDescriptor {
    return .{
        .name = name,
        .version = .{ .major = 1, .minor = 2, .patch = 3 },
        .author = "integration-test",
        .description = "test plugin",
        .capabilities = caps,
        .abi_version = PluginApi.current_version,
    };
}

// ── Registration ───────────────────────────────────────────────────────

test "plugin registry: register valid plugin appears in list" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const caps = [_]PluginCapability{ .ai_provider, .inference_engine };
    try reg.register(std.testing.allocator, makeDescriptor("test-ai-backend", &caps), .{});

    try std.testing.expectEqual(@as(usize, 1), reg.count());

    // Verify via get
    const entry = reg.get("test-ai-backend") orelse
        return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("test-ai-backend", entry.descriptor.name);
    try std.testing.expectEqual(PluginState.registered, entry.state);

    // Verify via list
    const descriptors = try reg.list(std.testing.allocator);
    defer std.testing.allocator.free(descriptors);
    try std.testing.expectEqual(@as(usize, 1), descriptors.len);
    try std.testing.expectEqualStrings("test-ai-backend", descriptors[0].name);
}

test "plugin registry: register duplicate returns PluginAlreadyRegistered" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const desc = makeDescriptor("dup-plugin", &.{});
    try reg.register(std.testing.allocator, desc, .{});

    try std.testing.expectError(
        PluginError.PluginAlreadyRegistered,
        reg.register(std.testing.allocator, desc, .{}),
    );
    // Count must remain 1
    try std.testing.expectEqual(@as(usize, 1), reg.count());
}

test "plugin registry: register incompatible ABI version returns error" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const bad_desc = PluginDescriptor{
        .name = "bad-abi-plugin",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .author = "test",
        .description = "incompatible",
        .capabilities = &.{},
        .abi_version = .{ .major = PluginApi.current_version.major + 99, .minor = 0, .patch = 0 },
    };

    try std.testing.expectError(
        PluginError.IncompatibleAbiVersion,
        reg.register(std.testing.allocator, bad_desc, .{}),
    );
    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

test "plugin registry: register with empty name returns InvalidPluginName" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expectError(
        PluginError.InvalidPluginName,
        reg.register(std.testing.allocator, .{
            .name = "",
            .author = "",
            .description = "",
        }, .{}),
    );
}

// ── Lifecycle: load ────────────────────────────────────────────────────

test "plugin registry: load transitions registered to active" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);
    resetCounters();

    try reg.register(std.testing.allocator, makeDescriptor("loadable", &.{}), .{
        .on_load = &countingOnLoad,
        .on_unload = &countingOnUnload,
    });

    // Pre-load state
    try std.testing.expectEqual(PluginState.registered, reg.get("loadable").?.state);
    try std.testing.expectEqual(@as(u32, 0), load_counter);

    // Load
    try reg.load("loadable");

    try std.testing.expectEqual(PluginState.active, reg.get("loadable").?.state);
    try std.testing.expectEqual(@as(u32, 1), load_counter);
}

test "plugin registry: load already active returns PluginStateInvalid" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, makeDescriptor("double-load", &.{}), .{});
    try reg.load("double-load");

    try std.testing.expectError(
        PluginError.PluginStateInvalid,
        reg.load("double-load"),
    );
}

test "plugin registry: load nonexistent plugin returns PluginNotFound" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expectError(
        PluginError.PluginNotFound,
        reg.load("does-not-exist"),
    );
}

test "plugin registry: load with failing callback sets failed state" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);
    resetCounters();
    fail_on_load = true;

    try reg.register(std.testing.allocator, makeDescriptor("fail-load", &.{}), .{
        .on_load = &countingOnLoad,
    });

    try std.testing.expectError(PluginError.PluginLoadFailed, reg.load("fail-load"));
    try std.testing.expectEqual(PluginState.failed, reg.get("fail-load").?.state);
}

// ── Lifecycle: unload ──────────────────────────────────────────────────

test "plugin registry: unload active plugin fires on_unload callback" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);
    resetCounters();

    try reg.register(std.testing.allocator, makeDescriptor("unloadable", &.{}), .{
        .on_load = &countingOnLoad,
        .on_unload = &countingOnUnload,
    });

    try reg.load("unloadable");
    try std.testing.expectEqual(@as(u32, 0), unload_counter);

    try reg.unload("unloadable");
    try std.testing.expectEqual(@as(u32, 1), unload_counter);
    try std.testing.expectEqual(PluginState.registered, reg.get("unloadable").?.state);
}

test "plugin registry: unload non-active plugin returns PluginStateInvalid" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, makeDescriptor("not-active", &.{}), .{});

    // Still in registered state
    try std.testing.expectError(
        PluginError.PluginStateInvalid,
        reg.unload("not-active"),
    );
}

test "plugin registry: unload nonexistent plugin returns PluginNotFound" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expectError(
        PluginError.PluginNotFound,
        reg.unload("ghost"),
    );
}

// ── Capability queries ─────────────────────────────────────────────────

test "plugin registry: query plugins by capability" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const ai_caps = [_]PluginCapability{ .ai_provider, .inference_engine };
    const store_caps = [_]PluginCapability{.storage_backend};
    const multi_caps = [_]PluginCapability{ .ai_provider, .storage_backend, .cache_backend };

    try reg.register(std.testing.allocator, makeDescriptor("ai-plug", &ai_caps), .{});
    try reg.register(std.testing.allocator, makeDescriptor("store-plug", &store_caps), .{});
    try reg.register(std.testing.allocator, makeDescriptor("multi-plug", &multi_caps), .{});

    // ai_provider: ai-plug + multi-plug = 2
    const ai_list = try reg.listByCapability(.ai_provider, std.testing.allocator);
    defer std.testing.allocator.free(ai_list);
    try std.testing.expectEqual(@as(usize, 2), ai_list.len);
    try std.testing.expectEqual(@as(usize, 2), reg.countByCapability(.ai_provider));

    // storage_backend: store-plug + multi-plug = 2
    const store_list = try reg.listByCapability(.storage_backend, std.testing.allocator);
    defer std.testing.allocator.free(store_list);
    try std.testing.expectEqual(@as(usize, 2), store_list.len);

    // inference_engine: ai-plug only = 1
    const inf_list = try reg.listByCapability(.inference_engine, std.testing.allocator);
    defer std.testing.allocator.free(inf_list);
    try std.testing.expectEqual(@as(usize, 1), inf_list.len);

    // cache_backend: multi-plug only = 1
    try std.testing.expectEqual(@as(usize, 1), reg.countByCapability(.cache_backend));

    // gpu_backend: nobody = 0
    const gpu_list = try reg.listByCapability(.gpu_backend, std.testing.allocator);
    defer std.testing.allocator.free(gpu_list);
    try std.testing.expectEqual(@as(usize, 0), gpu_list.len);
    try std.testing.expectEqual(@as(usize, 0), reg.countByCapability(.gpu_backend));
}

// ── Info snapshots ─────────────────────────────────────────────────────

test "plugin registry: getInfo returns snapshot with correct state" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const caps = [_]PluginCapability{.connector};
    try reg.register(std.testing.allocator, makeDescriptor("info-target", &caps), .{});

    // Snapshot before load
    const info_before = reg.getInfo("info-target") orelse
        return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("info-target", info_before.name);
    try std.testing.expectEqual(PluginState.registered, info_before.state);
    try std.testing.expectEqual(@as(usize, 1), info_before.capabilities.len);

    // Load and check again
    try reg.load("info-target");
    const info_after = reg.getInfo("info-target") orelse
        return error.TestUnexpectedResult;
    try std.testing.expectEqual(PluginState.active, info_after.state);
}

test "plugin registry: getInfo returns null for unknown plugin" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expect(reg.getInfo("nonexistent") == null);
}

test "plugin registry: listInfo returns all snapshots" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, makeDescriptor("snap-a", &.{}), .{});
    try reg.register(std.testing.allocator, makeDescriptor("snap-b", &.{}), .{});

    const infos = try reg.listInfo(std.testing.allocator);
    defer std.testing.allocator.free(infos);

    try std.testing.expectEqual(@as(usize, 2), infos.len);
}

// ── Full lifecycle cycle ───────────────────────────────────────────────

test "plugin registry: full register-load-unload-re-register cycle" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);
    resetCounters();

    const caps = [_]PluginCapability{.ai_provider};

    // 1. Register
    try reg.register(std.testing.allocator, makeDescriptor("lifecycle-plug", &caps), .{
        .on_load = &countingOnLoad,
        .on_unload = &countingOnUnload,
    });
    try std.testing.expectEqual(@as(usize, 1), reg.count());
    try std.testing.expectEqual(PluginState.registered, reg.get("lifecycle-plug").?.state);

    // 2. Load
    try reg.load("lifecycle-plug");
    try std.testing.expectEqual(PluginState.active, reg.get("lifecycle-plug").?.state);
    try std.testing.expectEqual(@as(u32, 1), load_counter);

    // 3. Unload
    try reg.unload("lifecycle-plug");
    try std.testing.expectEqual(PluginState.registered, reg.get("lifecycle-plug").?.state);
    try std.testing.expectEqual(@as(u32, 1), unload_counter);

    // 4. Unregister
    try reg.unregister("lifecycle-plug");
    try std.testing.expectEqual(@as(usize, 0), reg.count());
    try std.testing.expect(reg.get("lifecycle-plug") == null);
    // unload_counter stays at 1 — unregister of a registered (not active) plugin
    // should NOT fire on_unload again.
    try std.testing.expectEqual(@as(u32, 1), unload_counter);

    // 5. Re-register with same name succeeds
    try reg.register(std.testing.allocator, makeDescriptor("lifecycle-plug", &caps), .{
        .on_load = &countingOnLoad,
        .on_unload = &countingOnUnload,
    });
    try std.testing.expectEqual(@as(usize, 1), reg.count());
    try std.testing.expectEqual(PluginState.registered, reg.get("lifecycle-plug").?.state);

    // 6. Load again
    try reg.load("lifecycle-plug");
    try std.testing.expectEqual(PluginState.active, reg.get("lifecycle-plug").?.state);
    try std.testing.expectEqual(@as(u32, 2), load_counter);
}

// ── Multiple plugins ───────────────────────────────────────────────────

test "plugin registry: multiple plugins coexist independently" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, makeDescriptor("plug-alpha", &.{.connector}), .{});
    try reg.register(std.testing.allocator, makeDescriptor("plug-beta", &.{.storage_backend}), .{});
    try reg.register(std.testing.allocator, makeDescriptor("plug-gamma", &.{.gpu_backend}), .{});

    try std.testing.expectEqual(@as(usize, 3), reg.count());

    // Load only one
    try reg.load("plug-beta");
    try std.testing.expectEqual(PluginState.active, reg.get("plug-beta").?.state);
    try std.testing.expectEqual(PluginState.registered, reg.get("plug-alpha").?.state);
    try std.testing.expectEqual(PluginState.registered, reg.get("plug-gamma").?.state);

    // Unregister one
    try reg.unregister("plug-alpha");
    try std.testing.expectEqual(@as(usize, 2), reg.count());
    try std.testing.expect(reg.get("plug-alpha") == null);
}

// ── Unregister active plugin fires on_unload ───────────────────────────

test "plugin registry: unregister active plugin fires on_unload" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);
    resetCounters();

    try reg.register(std.testing.allocator, makeDescriptor("active-unreg", &.{}), .{
        .on_load = &countingOnLoad,
        .on_unload = &countingOnUnload,
    });
    try reg.load("active-unreg");
    try std.testing.expectEqual(@as(u32, 1), load_counter);
    try std.testing.expectEqual(@as(u32, 0), unload_counter);

    // Unregister while active should invoke on_unload
    try reg.unregister("active-unreg");
    try std.testing.expectEqual(@as(u32, 1), unload_counter);
    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

// ── Failed plugin can retry load ───────────────────────────────────────

test "plugin registry: failed plugin can retry load successfully" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);
    resetCounters();
    fail_on_load = true;

    try reg.register(std.testing.allocator, makeDescriptor("retry-plug", &.{}), .{
        .on_load = &countingOnLoad,
    });

    // First attempt fails
    try std.testing.expectError(PluginError.PluginLoadFailed, reg.load("retry-plug"));
    try std.testing.expectEqual(PluginState.failed, reg.get("retry-plug").?.state);

    // Fix the issue and retry
    fail_on_load = false;
    try reg.load("retry-plug");
    try std.testing.expectEqual(PluginState.active, reg.get("retry-plug").?.state);
}

// ── Capability enum names ──────────────────────────────────────────────

test "plugin registry: PluginCapability.name returns tag string" {
    try std.testing.expectEqualStrings("ai_provider", PluginCapability.ai_provider.name());
    try std.testing.expectEqualStrings("connector", PluginCapability.connector.name());
    try std.testing.expectEqualStrings("storage_backend", PluginCapability.storage_backend.name());
    try std.testing.expectEqualStrings("gpu_backend", PluginCapability.gpu_backend.name());
    try std.testing.expectEqualStrings("custom", PluginCapability.custom.name());
}

// ── Version descriptor ─────────────────────────────────────────────────

test "plugin registry: descriptor version is preserved" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const desc = PluginDescriptor{
        .name = "versioned",
        .version = .{ .major = 3, .minor = 7, .patch = 11 },
        .author = "ver-author",
        .description = "version check",
        .capabilities = &.{},
        .abi_version = PluginApi.current_version,
    };
    try reg.register(std.testing.allocator, desc, .{});

    const entry = reg.get("versioned") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 3), entry.descriptor.version.major);
    try std.testing.expectEqual(@as(u32, 7), entry.descriptor.version.minor);
    try std.testing.expectEqual(@as(u32, 11), entry.descriptor.version.patch);

    // Also via getInfo
    const info = reg.getInfo("versioned") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 3), info.version.major);
}

// ── No-op callbacks are safe ───────────────────────────────────────────

test "plugin registry: null callbacks are safe through full lifecycle" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, makeDescriptor("null-cbs", &.{}), .{
        .on_load = null,
        .on_unload = null,
    });

    try reg.load("null-cbs");
    try std.testing.expectEqual(PluginState.active, reg.get("null-cbs").?.state);

    try reg.unload("null-cbs");
    try std.testing.expectEqual(PluginState.registered, reg.get("null-cbs").?.state);
}

// ── deinit calls on_unload for active plugins ──────────────────────────

test "plugin registry: deinit fires on_unload for active plugins" {
    resetCounters();

    var reg = PluginRegistry.init();
    try reg.register(std.testing.allocator, makeDescriptor("deinit-active", &.{}), .{
        .on_load = &countingOnLoad,
        .on_unload = &countingOnUnload,
    });
    try reg.load("deinit-active");
    try std.testing.expectEqual(@as(u32, 0), unload_counter);

    // deinit should fire on_unload for the active plugin
    reg.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), unload_counter);
}

// ── refAllDecls ────────────────────────────────────────────────────────

test {
    std.testing.refAllDecls(@This());
}
