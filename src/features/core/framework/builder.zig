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
    return self;
}

pub fn build(comptime Framework: type, comptime Builder: type, self: *Builder) Framework.Error!Framework {
    var config = self.config_builder.build();
    try config_module.validate(config);

    config.plugins = try config.plugins.dupe(self.allocator);
    if (self.io) |io| {
        return Framework.initWithIo(self.allocator, config, io);
    }
    return Framework.init(self.allocator, config);
}

test {
    std.testing.refAllDecls(@This());
}
