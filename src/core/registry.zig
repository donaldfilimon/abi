//! Core Registry System
const std = @import("std");
const sync = @import("../foundation/sync.zig");

/// A slash-command provided by a plugin.
pub const PluginCommand = struct {
    name: []const u8,
    summary: []const u8 = "",
    aliases: []const []const u8 = &.{},
};

/// A context provider declared by a plugin to augment the REPL prompt
/// context at startup. Each provider is called with `__context__:<name>`
/// via the plugin's `run()` entry point.
pub const ContextProvider = struct {
    name: []const u8,
    summary: []const u8 = "",
};

pub const PluginDescriptor = struct {
    name: []const u8,
    version: []const u8 = "",
    description: []const u8,
    target_feature: []const u8 = "",
    entry_point: []const u8 = "",
    /// Optional slash-commands registered by this plugin.
    commands: []const PluginCommand = &.{},
    /// Optional context providers that augment the REPL prompt context.
    context_providers: []const ContextProvider = &.{},
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    modules: std.StringArrayHashMapUnmanaged(PluginDescriptor),
    lock: sync.RwLock = .{},

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
            .modules = .{},
        };
    }

    pub fn deinit(self: *Registry) void {
        self.lock.lockWrite();
        defer self.lock.unlockWrite();
        var it = self.modules.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            freeDescriptor(self.allocator, entry.value_ptr.*);
        }
        self.modules.deinit(self.allocator);
    }

    pub fn register(self: *Registry, name: []const u8, info: []const u8) !void {
        try self.registerPlugin(.{
            .name = name,
            .description = info,
        });
    }

    pub fn registerPlugin(self: *Registry, descriptor: PluginDescriptor) !void {
        self.lock.lockWrite();
        defer self.lock.unlockWrite();

        const owned_name = try self.allocator.dupe(u8, descriptor.name);
        errdefer self.allocator.free(owned_name);

        const owned_descriptor = try dupeDescriptor(self.allocator, descriptor);
        errdefer freeDescriptor(self.allocator, owned_descriptor);

        const result = try self.modules.getOrPut(self.allocator, owned_name);
        if (result.found_existing) {
            self.allocator.free(owned_name);
            freeDescriptor(self.allocator, result.value_ptr.*);
            result.value_ptr.* = owned_descriptor;
            return;
        }
        result.key_ptr.* = owned_name;
        result.value_ptr.* = owned_descriptor;
    }

    pub fn loadPlugins(self: *Registry) !void {
        const plugin_registry = @import("../plugin_registry.zig");
        try plugin_registry.registerPlugins(self);
    }

    pub fn pluginCount(self: *Registry) usize {
        self.lock.lockRead();
        defer self.lock.unlockRead();
        return self.modules.count();
    }

    pub fn getPlugin(self: *Registry, name: []const u8) ?PluginDescriptor {
        self.lock.lockRead();
        defer self.lock.unlockRead();
        return self.modules.get(name);
    }

    pub fn appendPluginNames(self: *Registry, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged([]const u8)) !void {
        self.lock.lockRead();
        defer self.lock.unlockRead();
        var it = self.modules.iterator();
        while (it.next()) |entry| {
            try out.append(allocator, entry.value_ptr.name);
        }
    }

    pub fn snapshotPluginNames(self: *Registry, allocator: std.mem.Allocator) ![][]const u8 {
        self.lock.lockRead();
        defer self.lock.unlockRead();

        const names = try allocator.alloc([]const u8, self.modules.count());
        errdefer allocator.free(names);

        var filled: usize = 0;
        errdefer {
            for (names[0..filled]) |name| allocator.free(name);
        }

        var it = self.modules.iterator();
        while (it.next()) |entry| {
            names[filled] = try allocator.dupe(u8, entry.value_ptr.name);
            filled += 1;
        }

        return names;
    }

    pub fn freePluginNamesSnapshot(allocator: std.mem.Allocator, names: [][]const u8) void {
        for (names) |name| allocator.free(name);
        allocator.free(names);
    }

    pub fn snapshotPlugins(self: *Registry, allocator: std.mem.Allocator) ![]PluginDescriptor {
        self.lock.lockRead();
        defer self.lock.unlockRead();

        const plugins = try allocator.alloc(PluginDescriptor, self.modules.count());
        errdefer allocator.free(plugins);

        var filled: usize = 0;
        errdefer {
            for (plugins[0..filled]) |plugin| freeDescriptor(allocator, plugin);
        }

        var it = self.modules.iterator();
        while (it.next()) |entry| {
            plugins[filled] = try dupeDescriptor(allocator, entry.value_ptr.*);
            filled += 1;
        }

        return plugins;
    }

    pub fn freePluginSnapshot(allocator: std.mem.Allocator, plugins: []PluginDescriptor) void {
        for (plugins) |plugin| freeDescriptor(allocator, plugin);
        allocator.free(plugins);
    }

    /// Find a plugin command by name (checking aliases too). Returns the owning
    /// plugin name and the command descriptor, or null if not found.
    pub fn findPluginCommand(self: *Registry, name: []const u8) ?struct { plugin: []const u8, command: PluginCommand } {
        self.lock.lockRead();
        defer self.lock.unlockRead();
        var it = self.modules.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.commands) |cmd| {
                if (std.mem.eql(u8, cmd.name, name)) return .{ .plugin = entry.key_ptr.*, .command = cmd };
                for (cmd.aliases) |alias| {
                    if (std.mem.eql(u8, alias, name)) return .{ .plugin = entry.key_ptr.*, .command = cmd };
                }
            }
        }
        return null;
    }

    pub fn formatPluginList(self: *Registry, allocator: std.mem.Allocator) ![]u8 {
        self.lock.lockRead();
        defer self.lock.unlockRead();

        var out: std.ArrayListUnmanaged(u8) = .empty;
        errdefer out.deinit(allocator);
        try out.print(allocator, "Installed Plugins ({d}):\n", .{self.modules.count()});

        var it = self.modules.iterator();
        while (it.next()) |entry| {
            const plugin = entry.value_ptr.*;
            if (plugin.version.len > 0 and plugin.target_feature.len > 0 and plugin.entry_point.len > 0) {
                try out.print(allocator, "  - {s} v{s} [{s}] ({s}): {s}\n", .{ plugin.name, plugin.version, plugin.target_feature, plugin.entry_point, plugin.description });
            } else if (plugin.version.len > 0 and plugin.target_feature.len > 0) {
                try out.print(allocator, "  - {s} v{s} [{s}]: {s}\n", .{ plugin.name, plugin.version, plugin.target_feature, plugin.description });
            } else {
                try out.print(allocator, "  - {s}: {s}\n", .{ plugin.name, plugin.description });
            }
        }

        return try out.toOwnedSlice(allocator);
    }
};

fn dupeDescriptor(allocator: std.mem.Allocator, descriptor: PluginDescriptor) !PluginDescriptor {
    const name = try allocator.dupe(u8, descriptor.name);
    errdefer allocator.free(name);
    const version = try allocator.dupe(u8, descriptor.version);
    errdefer allocator.free(version);
    const description = try allocator.dupe(u8, descriptor.description);
    errdefer allocator.free(description);
    const target_feature = try allocator.dupe(u8, descriptor.target_feature);
    errdefer allocator.free(target_feature);
    const entry_point = try allocator.dupe(u8, descriptor.entry_point);
    errdefer allocator.free(entry_point);

    const commands = try allocator.alloc(PluginCommand, descriptor.commands.len);
    errdefer allocator.free(commands);
    for (descriptor.commands, 0..) |cmd, i| {
        const cmd_name = try allocator.dupe(u8, cmd.name);
        errdefer allocator.free(cmd_name);
        const cmd_summary = try allocator.dupe(u8, cmd.summary);
        errdefer allocator.free(cmd_summary);
        const aliases = try allocator.alloc([]const u8, cmd.aliases.len);
        errdefer allocator.free(aliases);
        for (cmd.aliases, 0..) |alias, j| {
            aliases[j] = try allocator.dupe(u8, alias);
        }
        commands[i] = .{ .name = cmd_name, .summary = cmd_summary, .aliases = aliases };
    }

    const context_providers = try allocator.alloc(ContextProvider, descriptor.context_providers.len);
    errdefer allocator.free(context_providers);
    for (descriptor.context_providers, 0..) |cp, i| {
        const cp_name = try allocator.dupe(u8, cp.name);
        errdefer allocator.free(cp_name);
        const cp_summary = try allocator.dupe(u8, cp.summary);
        errdefer allocator.free(cp_summary);
        context_providers[i] = .{ .name = cp_name, .summary = cp_summary };
    }

    return .{
        .name = name,
        .version = version,
        .description = description,
        .target_feature = target_feature,
        .entry_point = entry_point,
        .commands = commands,
        .context_providers = context_providers,
    };
}

fn freeDescriptor(allocator: std.mem.Allocator, descriptor: PluginDescriptor) void {
    allocator.free(descriptor.name);
    allocator.free(descriptor.version);
    allocator.free(descriptor.description);
    allocator.free(descriptor.target_feature);
    allocator.free(descriptor.entry_point);
    for (descriptor.commands) |cmd| {
        allocator.free(cmd.name);
        allocator.free(cmd.summary);
        for (cmd.aliases) |alias| allocator.free(alias);
        allocator.free(cmd.aliases);
    }
    allocator.free(descriptor.commands);
    for (descriptor.context_providers) |cp| {
        allocator.free(cp.name);
        allocator.free(cp.summary);
    }
    allocator.free(descriptor.context_providers);
}

pub const Config = struct {
    max_concurrent_streams: u32 = 10,
    heartbeat_interval_ms: u32 = 5000,
    default_backend: []const u8 = "stdgpu",
};

test {
    std.testing.refAllDecls(@This());
}

test "Registry owns registered entries" {
    const testing = std.testing;

    var registry = Registry.init(testing.allocator);
    defer registry.deinit();

    try registry.register("example", "first");
    try registry.register("example", "second");

    const descriptor = registry.getPlugin("example") orelse return error.MissingPlugin;
    try testing.expectEqualStrings("second", descriptor.description);
    try testing.expectEqual(@as(usize, 1), registry.pluginCount());
}

test "Registry stores generated plugin metadata" {
    const testing = std.testing;

    var registry = Registry.init(testing.allocator);
    defer registry.deinit();

    try registry.registerPlugin(.{
        .name = "example",
        .version = "1.2.3",
        .description = "metadata test",
        .target_feature = "ai",
        .entry_point = "mod.zig",
    });

    const descriptor = registry.getPlugin("example") orelse return error.MissingPlugin;
    try testing.expectEqualStrings("1.2.3", descriptor.version);
    try testing.expectEqualStrings("metadata test", descriptor.description);
    try testing.expectEqualStrings("ai", descriptor.target_feature);
    try testing.expectEqualStrings("mod.zig", descriptor.entry_point);
}

test "Registry snapshots plugin names with owned memory" {
    const testing = std.testing;

    var registry = Registry.init(testing.allocator);
    defer registry.deinit();

    try registry.registerPlugin(.{
        .name = "example",
        .version = "1.2.3",
        .description = "metadata test",
        .target_feature = "ai",
        .entry_point = "mod.zig",
    });

    const names = try registry.snapshotPluginNames(testing.allocator);
    defer Registry.freePluginNamesSnapshot(testing.allocator, names);
    try testing.expectEqual(@as(usize, 1), names.len);
    try testing.expectEqualStrings("example", names[0]);
}

test "Registry snapshots complete plugin descriptors with owned memory" {
    const testing = std.testing;

    var registry = Registry.init(testing.allocator);
    defer registry.deinit();

    try registry.registerPlugin(.{
        .name = "example",
        .version = "1.2.3",
        .description = "metadata test",
        .target_feature = "ai",
        .entry_point = "mod.zig",
    });

    const plugins = try registry.snapshotPlugins(testing.allocator);
    defer Registry.freePluginSnapshot(testing.allocator, plugins);
    try testing.expectEqual(@as(usize, 1), plugins.len);
    try testing.expectEqualStrings("example", plugins[0].name);
    try testing.expectEqualStrings("1.2.3", plugins[0].version);
    try testing.expectEqualStrings("metadata test", plugins[0].description);
    try testing.expectEqualStrings("ai", plugins[0].target_feature);
    try testing.expectEqualStrings("mod.zig", plugins[0].entry_point);
}

test "Registry formats plugin list with metadata" {
    const testing = std.testing;

    var registry = Registry.init(testing.allocator);
    defer registry.deinit();

    try registry.registerPlugin(.{
        .name = "example",
        .version = "1.2.3",
        .description = "metadata test",
        .target_feature = "ai",
        .entry_point = "mod.zig",
    });

    const rendered = try registry.formatPluginList(testing.allocator);
    defer testing.allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "Installed Plugins (1):") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "example v1.2.3 [ai] (mod.zig)") != null);
}
