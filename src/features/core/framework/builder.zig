//! Builder logic split out of the framework façade.

const std = @import("std");
const config_module = @import("../config/mod.zig");
const shutdown = @import("shutdown.zig");

pub fn init(comptime Builder: type, allocator: std.mem.Allocator) Builder {
    return .{
        .allocator = allocator,
        .config_builder = config_module.Builder.init(allocator),
        .io = null,
        .plugins = .empty,
    };
}

pub fn withDefaults(comptime Builder: type, self: *Builder) *Builder {
    _ = self.config_builder.withDefaults();
    return self;
}

pub fn withIo(comptime Builder: type, self: *Builder, io: std.Io) *Builder {
    self.io = io;
    return self;
}

/// Generic feature configuration setter — delegates to config_builder.with().
pub fn with(comptime Builder: type, self: *Builder, comptime feature: config_module.Feature, cfg: anytype) *Builder {
    _ = self.config_builder.with(feature, cfg);
    return self;
}

/// Generic feature defaults setter — delegates to config_builder.withDefault().
pub fn withDefault(comptime Builder: type, self: *Builder, comptime feature: config_module.Feature) *Builder {
    _ = self.config_builder.withDefault(feature);
    return self;
}

pub fn withPlugins(comptime Builder: type, self: *Builder, plugin_cfg: config_module.PluginConfig) *Builder {
    self.config_builder.config.plugins = plugin_cfg;
<<<<<<< Updated upstream:src/features/core/framework/builder.zig
=======
    return self;
}

pub fn registerPlugin(comptime Builder: type, self: *Builder, plugin: config_module.plugin_config.Plugin) *Builder {
    self.plugins.append(self.allocator, plugin) catch {
        self.plugins_oom = true;
    };
    return self;
}

pub fn registerDynLibPlugin(comptime Builder: type, self: *Builder, lib: std.DynLib) *Builder {
    self.plugins.append(self.allocator, .{ .dyn_lib = lib }) catch {
        self.plugins_oom = true;
    };
    return self;
}

pub fn registerStaticPlugin(comptime Builder: type, self: *Builder, ptr: ?*anyopaque, init_fn: *const fn (ptr: ?*anyopaque, fw: *anyopaque) anyerror!void) *Builder {
    self.plugins.append(self.allocator, .{ .static = .{
        .ptr = ptr,
        .init_plugin = init_fn,
    } }) catch {
        self.plugins_oom = true;
    };
>>>>>>> Stashed changes:src/core/framework/builder.zig
    return self;
}

pub fn build(comptime Framework: type, comptime Builder: type, self: *Builder) Framework.Error!Framework {
<<<<<<< Updated upstream:src/features/core/framework/builder.zig
    var config = self.config_builder.build();
    try config_module.validate(config);

    config.plugins = try config.plugins.dupe(self.allocator);
    if (self.io) |io| {
        return Framework.initWithIo(self.allocator, config, io);
=======
    defer {
        self.plugins.deinit(self.allocator);
        self.plugins = .empty;
>>>>>>> Stashed changes:src/core/framework/builder.zig
    }

    if (self.plugins_oom) {
        shutdown.deinitOwnedPlugins(self.allocator, self.plugins.items);
        self.plugins = .empty;
        return error.OutOfMemory;
    }

    var config = self.config_builder.build();
    const combined_len = config.plugins.plugins.len + self.plugins.items.len;
    var owned_plugins: ?[]config_module.plugin_config.Plugin = null;
    if (combined_len > 0) {
        const slice = try self.allocator.alloc(config_module.plugin_config.Plugin, combined_len);
        errdefer shutdown.deinitOwnedPlugins(self.allocator, slice);

        const preconfigured_len = config.plugins.plugins.len;
        @memcpy(slice[0..preconfigured_len], config.plugins.plugins);
        @memcpy(slice[preconfigured_len..], self.plugins.items);

        owned_plugins = slice;
        config.plugins.plugins = slice;
    } else {
        config.plugins.plugins = &[_]config_module.plugin_config.Plugin{};
    }

    var framework = if (self.io) |io|
        try Framework.initWithIo(self.allocator, config, io)
    else
        try Framework.init(self.allocator, config);
    framework.owned_plugins = owned_plugins;
    return framework;
}

test {
    std.testing.refAllDecls(@This());
}
