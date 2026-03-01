//! Builder logic split out of the framework façade.

const std = @import("std");
const config_module = @import("../config/mod.zig");

pub fn init(comptime Builder: type, allocator: std.mem.Allocator) Builder {
    return .{
        .allocator = allocator,
        .config_builder = config_module.Builder.init(allocator),
        .io = null,
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
    _ = self.config_builder.with(.plugins, plugin_cfg);
    return self;
}

pub fn build(comptime Framework: type, comptime Builder: type, self: *Builder) Framework.Error!Framework {
    const config = self.config_builder.build();
    if (self.io) |io| {
        return Framework.initWithIo(self.allocator, config, io);
    }
    return Framework.init(self.allocator, config);
}

test {
    std.testing.refAllDecls(@This());
}
