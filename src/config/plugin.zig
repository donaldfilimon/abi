//! Plugin Configuration
//!
//! Configuration types for plugin loading and management.

/// Plugin loading and discovery settings.
pub const PluginConfig = struct {
    /// Paths to search for plugins.
    paths: []const []const u8 = &[_][]const u8{},

    /// Auto-discover plugins in paths.
    auto_discover: bool = false,

    /// Plugins to load by name.
    load: []const []const u8 = &[_][]const u8{},

    /// Allow loading untrusted plugins.
    allow_untrusted: bool = false,

    pub fn defaults() PluginConfig {
        return .{};
    }

    /// Configuration with specific plugin paths.
    pub fn withPaths(paths: []const []const u8) PluginConfig {
        return .{
            .paths = paths,
            .auto_discover = true,
        };
    }
};
