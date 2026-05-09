//! Plugin Ecosystem
//!
//! Provides a registry for external plugins (C/WASM shared libraries, FFI modules)
//! that extend framework capabilities at runtime. Plugins declare their capabilities
//! via `PluginDescriptor` and provide lifecycle callbacks via `PluginCallbacks`.
//!
//! ## Usage
//!
//! ```zig
//! const plugin_mod = @import("registry").plugin;
//!
//! var reg = plugin_mod.PluginRegistry.init(allocator);
//! defer reg.deinit();
//!
//! try reg.register(allocator, .{
//!     .name = "my-ai-backend",
//!     .version = .{ .major = 1, .minor = 0, .patch = 0 },
//!     .author = "acme",
//!     .description = "Custom AI inference backend",
//!     .capabilities = &.{ .ai_provider, .inference_engine },
//!     .abi_version = types.PluginApi.current_version,
//! }, .{});
//!
//! try reg.load("my-ai-backend");
//! ```

const std = @import("std");
const types = @import("types.zig");

const PluginApi = types.PluginApi;

// ============================================================================
// Plugin Types
// ============================================================================

/// What framework subsystem a plugin can extend.
pub const PluginCapability = enum {
    /// Provides AI model inference (e.g. custom LLM backend).
    ai_provider,
    /// Provides a data connector (HTTP, gRPC, etc.).
    connector,
    /// Provides a storage backend (S3-compatible, local FS, etc.).
    storage_backend,
    /// Provides a GPU compute backend.
    gpu_backend,
    /// Provides an inference engine (ONNX, TensorRT, etc.).
    inference_engine,
    /// Provides a vector index implementation.
    vector_index,
    /// Provides an authentication / authorization provider.
    auth_provider,
    /// Provides a cache backend (Redis, Memcached, etc.).
    cache_backend,
    /// Provides a search engine implementation.
    search_engine,
    /// Provides a message broker (Kafka, NATS, etc.).
    message_broker,
    /// Arbitrary user-defined capability.
    custom,

    pub fn name(self: PluginCapability) []const u8 {
        return @tagName(self);
    }
};

/// Runtime state of a registered plugin.
pub const PluginState = enum {
    /// Registered but not yet loaded.
    registered,
    /// Currently executing its on_load callback.
    loading,
    /// Loaded and operational.
    active,
    /// Currently executing its on_unload callback.
    unloading,
    /// A lifecycle callback returned an error.
    failed,
};

/// Metadata describing a plugin.  Passed to `register`.
pub const PluginDescriptor = struct {
    /// Unique plugin name (must be non-empty).
    name: []const u8,
    /// Semantic version of the plugin itself.
    version: PluginApi.Version = PluginApi.current_version,
    /// Author / organisation.
    author: []const u8 = "",
    /// Human-readable description.
    description: []const u8 = "",
    /// Set of capabilities this plugin provides.
    capabilities: []const PluginCapability = &.{},
    /// ABI version the plugin was built against.
    /// Major must match `PluginApi.current_version.major`.
    abi_version: PluginApi.Version = PluginApi.current_version,
};

/// Lifecycle callbacks supplied by the plugin author.
/// All fields are optional; `null` means "no-op".
pub const PluginCallbacks = struct {
    /// Called when the plugin is loaded (after registration).
    on_load: ?*const fn () anyerror!void = null,
    /// Called when the plugin is unloaded.
    on_unload: ?*const fn () void = null,
};

/// Read-only snapshot of a plugin's state for enumeration.
pub const PluginInfo = struct {
    name: []const u8,
    version: PluginApi.Version,
    author: []const u8,
    description: []const u8,
    state: PluginState,
    capabilities: []const PluginCapability,
};

/// A single plugin tracked inside the registry.
pub const PluginEntry = struct {
    descriptor: PluginDescriptor,
    state: PluginState,
    callbacks: PluginCallbacks,
};

// ============================================================================
// Error Set
// ============================================================================

/// Error set for plugin registry operations.
pub const PluginError = error{
    PluginAlreadyRegistered,
    PluginNotFound,
    PluginLoadFailed,
    PluginStateInvalid,
    IncompatibleAbiVersion,
    InvalidPluginName,
    OutOfMemory,
};

// ============================================================================
// Plugin Registry
// ============================================================================

/// Central registry managing plugin entries keyed by name.
pub const PluginRegistry = struct {
    entries: std.StringHashMapUnmanaged(PluginEntry),

    /// Create an empty plugin registry.
    pub fn init() PluginRegistry {
        return .{
            .entries = .empty,
        };
    }

    /// Tear down the registry, unloading every active plugin first.
    pub fn deinit(self: *PluginRegistry, allocator: std.mem.Allocator) void {
        // Invoke on_unload for any active plugins
        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            if (entry.state == .active) {
                if (entry.callbacks.on_unload) |on_unload| {
                    on_unload();
                }
            }
        }
        self.entries.deinit(allocator);
    }

    // ====================================================================
    // Registration
    // ====================================================================

    /// Register a plugin. Returns error if a plugin with the same name
    /// exists, the name is empty, or the ABI version is incompatible.
    pub fn register(
        self: *PluginRegistry,
        allocator: std.mem.Allocator,
        descriptor: PluginDescriptor,
        callbacks: PluginCallbacks,
    ) PluginError!void {
        if (descriptor.name.len == 0) return PluginError.InvalidPluginName;

        // ABI major-version gate
        if (descriptor.abi_version.major != PluginApi.current_version.major) {
            return PluginError.IncompatibleAbiVersion;
        }

        const gop = self.entries.getOrPut(allocator, descriptor.name) catch
            return PluginError.OutOfMemory;
        if (gop.found_existing) {
            return PluginError.PluginAlreadyRegistered;
        }
        gop.value_ptr.* = .{
            .descriptor = descriptor,
            .state = .registered,
            .callbacks = callbacks,
        };
    }

    /// Remove a plugin by name.  Invokes on_unload if active.
    pub fn unregister(self: *PluginRegistry, plugin_name: []const u8) PluginError!void {
        const entry_ptr = self.entries.getPtr(plugin_name) orelse
            return PluginError.PluginNotFound;

        // Invoke on_unload if active
        if (entry_ptr.state == .active) {
            entry_ptr.state = .unloading;
            if (entry_ptr.callbacks.on_unload) |on_unload| {
                on_unload();
            }
        }

        _ = self.entries.remove(plugin_name);
    }

    // ====================================================================
    // Lifecycle
    // ====================================================================

    /// Load a registered plugin, invoking its `on_load` callback.
    /// Only valid from `registered` or `failed` state.
    pub fn load(self: *PluginRegistry, plugin_name: []const u8) PluginError!void {
        const entry = self.entries.getPtr(plugin_name) orelse
            return PluginError.PluginNotFound;

        if (entry.state != .registered and entry.state != .failed) {
            return PluginError.PluginStateInvalid;
        }

        entry.state = .loading;

        if (entry.callbacks.on_load) |on_load| {
            on_load() catch {
                entry.state = .failed;
                return PluginError.PluginLoadFailed;
            };
        }

        entry.state = .active;
    }

    /// Unload an active plugin, invoking its `on_unload` callback.
    /// Transitions back to `registered` state.
    pub fn unload(self: *PluginRegistry, plugin_name: []const u8) PluginError!void {
        const entry = self.entries.getPtr(plugin_name) orelse
            return PluginError.PluginNotFound;

        if (entry.state != .active) {
            return PluginError.PluginStateInvalid;
        }

        entry.state = .unloading;

        if (entry.callbacks.on_unload) |on_unload| {
            on_unload();
        }

        entry.state = .registered;
    }

    // ====================================================================
    // Query
    // ====================================================================

    /// Look up a plugin entry by name.
    pub fn get(self: *PluginRegistry, plugin_name: []const u8) ?*PluginEntry {
        return self.entries.getPtr(plugin_name);
    }

    /// Look up a read-only snapshot by name.
    pub fn getInfo(self: *const PluginRegistry, plugin_name: []const u8) ?PluginInfo {
        const entry = self.entries.get(plugin_name) orelse return null;
        return infoFromEntry(entry);
    }

    /// Return a caller-owned slice of all registered plugin descriptors.
    pub fn list(self: *const PluginRegistry, allocator: std.mem.Allocator) PluginError![]const PluginDescriptor {
        var result = std.ArrayListUnmanaged(PluginDescriptor).empty;
        errdefer result.deinit(allocator);

        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            result.append(allocator, entry.descriptor) catch
                return PluginError.OutOfMemory;
        }
        return result.toOwnedSlice(allocator) catch return PluginError.OutOfMemory;
    }

    /// Return a caller-owned slice of plugin info snapshots.
    pub fn listInfo(self: *const PluginRegistry, allocator: std.mem.Allocator) PluginError![]PluginInfo {
        var result = std.ArrayListUnmanaged(PluginInfo).empty;
        errdefer result.deinit(allocator);

        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            result.append(allocator, infoFromEntry(entry.*)) catch
                return PluginError.OutOfMemory;
        }
        return result.toOwnedSlice(allocator) catch return PluginError.OutOfMemory;
    }

    /// Return names of plugins that declare a given capability.
    /// Caller owns the returned slice.
    pub fn listByCapability(
        self: *const PluginRegistry,
        capability: PluginCapability,
        allocator: std.mem.Allocator,
    ) PluginError![]const []const u8 {
        var result = std.ArrayListUnmanaged([]const u8).empty;
        errdefer result.deinit(allocator);

        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            for (entry.descriptor.capabilities) |cap| {
                if (cap == capability) {
                    result.append(allocator, entry.descriptor.name) catch
                        return PluginError.OutOfMemory;
                    break;
                }
            }
        }
        return result.toOwnedSlice(allocator) catch return PluginError.OutOfMemory;
    }

    /// Count plugins that advertise a given capability.
    pub fn countByCapability(self: *const PluginRegistry, capability: PluginCapability) usize {
        var n: usize = 0;
        var iter = self.entries.valueIterator();
        while (iter.next()) |entry| {
            for (entry.descriptor.capabilities) |cap| {
                if (cap == capability) {
                    n += 1;
                    break;
                }
            }
        }
        return n;
    }

    /// Total number of registered plugins.
    pub fn count(self: *const PluginRegistry) usize {
        return self.entries.count();
    }

    // ====================================================================
    // Internal helpers
    // ====================================================================

    fn infoFromEntry(entry: PluginEntry) PluginInfo {
        return .{
            .name = entry.descriptor.name,
            .version = entry.descriptor.version,
            .author = entry.descriptor.author,
            .description = entry.descriptor.description,
            .state = entry.state,
            .capabilities = entry.descriptor.capabilities,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "PluginRegistry init and deinit" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), reg.count());
}

test "register and get plugin" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const caps = [_]PluginCapability{ .ai_provider, .inference_engine };
    const desc = PluginDescriptor{
        .name = "test-plugin",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .author = "test-author",
        .description = "A test plugin",
        .capabilities = &caps,
        .abi_version = PluginApi.current_version,
    };

    try reg.register(std.testing.allocator, desc, .{});
    try std.testing.expectEqual(@as(usize, 1), reg.count());

    const entry = reg.get("test-plugin").?;
    try std.testing.expectEqualStrings("test-plugin", entry.descriptor.name);
    try std.testing.expectEqual(PluginState.registered, entry.state);
}

test "register duplicate returns error" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const desc = PluginDescriptor{
        .name = "dup",
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
        .author = "author",
        .description = "duplicate test",
        .capabilities = &.{},
        .abi_version = PluginApi.current_version,
    };

    try reg.register(std.testing.allocator, desc, .{});
    try std.testing.expectError(
        PluginError.PluginAlreadyRegistered,
        reg.register(std.testing.allocator, desc, .{}),
    );
}

test "register empty name returns error" {
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

test "register incompatible ABI version" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expectError(
        PluginError.IncompatibleAbiVersion,
        reg.register(std.testing.allocator, .{
            .name = "bad-abi",
            .author = "",
            .description = "",
            .abi_version = .{ .major = 999, .minor = 0, .patch = 0 },
        }, .{}),
    );
}

test "unregister plugin" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const desc = PluginDescriptor{
        .name = "removable",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .author = "author",
        .description = "to be removed",
        .capabilities = &.{},
        .abi_version = PluginApi.current_version,
    };

    try reg.register(std.testing.allocator, desc, .{});
    try std.testing.expectEqual(@as(usize, 1), reg.count());

    try reg.unregister("removable");
    try std.testing.expectEqual(@as(usize, 0), reg.count());
    try std.testing.expect(reg.get("removable") == null);
}

test "unregister missing plugin returns error" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expectError(
        PluginError.PluginNotFound,
        reg.unregister("nope"),
    );
}

test "load and unload plugin" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, .{
        .name = "loadable",
        .author = "",
        .description = "",
    }, .{});

    // Load transitions to active
    try reg.load("loadable");
    try std.testing.expectEqual(PluginState.active, reg.get("loadable").?.state);

    // Unload transitions back to registered
    try reg.unload("loadable");
    try std.testing.expectEqual(PluginState.registered, reg.get("loadable").?.state);
}

test "load already-active plugin returns error" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, .{
        .name = "active-test",
        .author = "",
        .description = "",
    }, .{});
    try reg.load("active-test");

    try std.testing.expectError(
        PluginError.PluginStateInvalid,
        reg.load("active-test"),
    );
}

test "list returns all descriptors" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const desc_a = PluginDescriptor{
        .name = "alpha",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .author = "a",
        .description = "first",
        .capabilities = &.{.connector},
        .abi_version = PluginApi.current_version,
    };
    const desc_b = PluginDescriptor{
        .name = "beta",
        .version = .{ .major = 2, .minor = 0, .patch = 0 },
        .author = "b",
        .description = "second",
        .capabilities = &.{.storage_backend},
        .abi_version = PluginApi.current_version,
    };

    try reg.register(std.testing.allocator, desc_a, .{});
    try reg.register(std.testing.allocator, desc_b, .{});

    const descriptors = try reg.list(std.testing.allocator);
    defer std.testing.allocator.free(descriptors);

    try std.testing.expectEqual(@as(usize, 2), descriptors.len);
}

test "listInfo returns snapshots" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, .{
        .name = "info-test",
        .author = "tester",
        .description = "info test",
    }, .{});

    const infos = try reg.listInfo(std.testing.allocator);
    defer std.testing.allocator.free(infos);

    try std.testing.expectEqual(@as(usize, 1), infos.len);
    try std.testing.expectEqualStrings("info-test", infos[0].name);
    try std.testing.expectEqual(PluginState.registered, infos[0].state);
}

test "getInfo returns null for unknown" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try std.testing.expect(reg.getInfo("nope") == null);
}

test "countByCapability" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    const caps_1 = [_]PluginCapability{ .ai_provider, .connector };
    const caps_2 = [_]PluginCapability{.ai_provider};

    try reg.register(std.testing.allocator, .{
        .name = "p1",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .author = "a",
        .description = "first",
        .capabilities = &caps_1,
        .abi_version = PluginApi.current_version,
    }, .{});

    try reg.register(std.testing.allocator, .{
        .name = "p2",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .author = "b",
        .description = "second",
        .capabilities = &caps_2,
        .abi_version = PluginApi.current_version,
    }, .{});

    try std.testing.expectEqual(@as(usize, 2), reg.countByCapability(.ai_provider));
    try std.testing.expectEqual(@as(usize, 1), reg.countByCapability(.connector));
    try std.testing.expectEqual(@as(usize, 0), reg.countByCapability(.gpu_backend));
}

test "listByCapability filters correctly" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    try reg.register(std.testing.allocator, .{
        .name = "ai-plug",
        .author = "",
        .description = "",
        .capabilities = &.{ .ai_provider, .inference_engine },
    }, .{});
    try reg.register(std.testing.allocator, .{
        .name = "store-plug",
        .author = "",
        .description = "",
        .capabilities = &.{.storage_backend},
    }, .{});
    try reg.register(std.testing.allocator, .{
        .name = "multi-plug",
        .author = "",
        .description = "",
        .capabilities = &.{ .ai_provider, .storage_backend },
    }, .{});

    const ai_list = try reg.listByCapability(.ai_provider, std.testing.allocator);
    defer std.testing.allocator.free(ai_list);
    try std.testing.expectEqual(@as(usize, 2), ai_list.len);

    const store_list = try reg.listByCapability(.storage_backend, std.testing.allocator);
    defer std.testing.allocator.free(store_list);
    try std.testing.expectEqual(@as(usize, 2), store_list.len);

    const gpu_list = try reg.listByCapability(.gpu_backend, std.testing.allocator);
    defer std.testing.allocator.free(gpu_list);
    try std.testing.expectEqual(@as(usize, 0), gpu_list.len);
}

var test_load_called: bool = false;
var test_unload_called: bool = false;

fn testOnLoad() anyerror!void {
    test_load_called = true;
}

fn testOnUnload() void {
    test_unload_called = true;
}

test "callbacks are invoked on load and unload" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    test_load_called = false;
    test_unload_called = false;

    try reg.register(std.testing.allocator, .{
        .name = "cb-test",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .author = "test",
        .description = "callback test",
        .capabilities = &.{},
        .abi_version = PluginApi.current_version,
    }, .{
        .on_load = &testOnLoad,
        .on_unload = &testOnUnload,
    });

    // After registration, plugin should be in registered state
    const entry = reg.get("cb-test").?;
    try std.testing.expectEqual(PluginState.registered, entry.state);
    try std.testing.expect(!test_load_called);

    // Load invokes on_load
    try reg.load("cb-test");
    try std.testing.expect(test_load_called);
    try std.testing.expectEqual(PluginState.active, reg.get("cb-test").?.state);

    // Unload invokes on_unload
    try reg.unload("cb-test");
    try std.testing.expect(test_unload_called);
    try std.testing.expectEqual(PluginState.registered, reg.get("cb-test").?.state);
}

var test_fail_load_called: bool = false;

fn testFailOnLoad() anyerror!void {
    test_fail_load_called = true;
    return error.TestLoadFailure;
}

test "load failure sets failed state" {
    var reg = PluginRegistry.init();
    defer reg.deinit(std.testing.allocator);

    test_fail_load_called = false;

    try reg.register(std.testing.allocator, .{
        .name = "fail-load",
        .author = "",
        .description = "",
    }, .{
        .on_load = &testFailOnLoad,
    });

    try std.testing.expectError(PluginError.PluginLoadFailed, reg.load("fail-load"));
    try std.testing.expect(test_fail_load_called);
    try std.testing.expectEqual(PluginState.failed, reg.get("fail-load").?.state);

    // Can retry from failed state
    test_fail_load_called = false;
    try std.testing.expectError(PluginError.PluginLoadFailed, reg.load("fail-load"));
    try std.testing.expect(test_fail_load_called);
}

test {
    std.testing.refAllDecls(@This());
}
