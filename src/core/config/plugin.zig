//! Plugin Configuration
//!
//! Configuration types for plugin loading and management.

const std = @import("std");

/// Plugin loading and discovery settings.
pub const PluginConfig = struct {
    /// Paths to search for plugins.
    paths: []const []const u8 = &[_][]const u8{},
    paths_owned: bool = false,

    /// Auto-discover plugins in paths.
    auto_discover: bool = false,

    /// Plugins to load by name.
    load: []const []const u8 = &[_][]const u8{},
    load_owned: bool = false,

    /// Allow loading untrusted plugins.
    allow_untrusted: bool = false,

    fn deinitStringSlices(allocator: std.mem.Allocator, slices: []const []const u8) void {
        var idx: usize = slices.len;
        while (idx > 0) : (idx -= 1) {
            allocator.free(slices[idx - 1]);
        }
        allocator.free(slices);
    }

    pub fn dupe(self: PluginConfig, allocator: std.mem.Allocator) !PluginConfig {
        const paths = try dupeStringSlices(allocator, self.paths);
        errdefer deinitStringSlices(allocator, paths);

        const load = try dupeStringSlices(allocator, self.load);
        errdefer deinitStringSlices(allocator, load);

        return .{
            .paths = paths,
            .paths_owned = true,
            .auto_discover = self.auto_discover,
            .load = load,
            .load_owned = true,
            .allow_untrusted = self.allow_untrusted,
        };
    }

    pub fn deinit(self: *PluginConfig, allocator: std.mem.Allocator) void {
        if (self.paths_owned) {
            deinitStringSlices(allocator, self.paths);
        }
        if (self.load_owned) {
            deinitStringSlices(allocator, self.load);
        }
        self.* = .{};
    }

    pub fn defaults() PluginConfig {
        return .{};
    }

    /// Configuration with specific plugin paths.
    pub fn withPaths(paths: []const []const u8) PluginConfig {
        return .{
            .paths = paths,
            .auto_discover = true,
            .allow_untrusted = true,
        };
    }
};

fn dupeStringSlices(allocator: std.mem.Allocator, slices: []const []const u8) ![]const []const u8 {
    const out = try allocator.alloc([]const u8, slices.len);
    errdefer allocator.free(out);

    var idx: usize = 0;
    errdefer {
        while (idx > 0) : (idx -= 1) {
            allocator.free(out[idx - 1]);
        }
    }

    while (idx < slices.len) : (idx += 1) {
        out[idx] = try allocator.dupe(u8, slices[idx]);
    }

    return out;
}

test "plugin config withPaths opts into untrusted loading" {
    const config = PluginConfig.withPaths(&.{"./plugins"});
    try std.testing.expect(config.auto_discover);
    try std.testing.expect(config.allow_untrusted);
    try std.testing.expectEqual(@as(usize, 1), config.paths.len);
}
