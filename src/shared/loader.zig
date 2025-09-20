//! Plugin Loader
//!
//! This module handles the cross-platform loading of plugin shared libraries
//! and provides a safe interface for plugin discovery and loading.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const interface = @import("interface.zig");

const PluginError = types.PluginError;
const PluginInterface = interface.PluginInterface;
const PluginFactoryFn = interface.PluginFactoryFn;
const PLUGIN_ENTRY_POINT = interface.PLUGIN_ENTRY_POINT;

/// Platform-specific library handle type
const LibraryHandle = switch (builtin.os.tag) {
    .windows => std.os.windows.HMODULE,
    .linux, .macos, .freebsd, .openbsd, .netbsd, .dragonfly => *anyopaque,
    else => *anyopaque,
};

/// Cross-platform plugin loader
pub const PluginLoader = struct {
    allocator: std.mem.Allocator,
    loaded_libraries: std.ArrayListUnmanaged(LoadedLibrary) = .{},
    plugin_paths: std.ArrayListUnmanaged([]u8) = .{},

    const LoadedLibrary = struct {
        path: []u8,
        handle: LibraryHandle,
        factory_fn: PluginFactoryFn,
        interface: ?*const PluginInterface = null,
    };

    pub fn init(allocator: std.mem.Allocator) PluginLoader {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *PluginLoader) void {
        // Unload all libraries
        for (self.loaded_libraries.items) |lib| {
            self.unloadLibrary(lib.handle) catch {};
            self.allocator.free(lib.path);
        }
        self.loaded_libraries.deinit(self.allocator);

        // Free plugin paths
        for (self.plugin_paths.items) |path| {
            self.allocator.free(path);
        }
        self.plugin_paths.deinit(self.allocator);
    }

    /// Add a directory to search for plugins
    pub fn addPluginPath(self: *PluginLoader, path: []const u8) !void {
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);
        try self.plugin_paths.append(self.allocator, owned_path);
    }

    /// Remove a plugin search path
    pub fn removePluginPath(self: *PluginLoader, path: []const u8) void {
        for (self.plugin_paths.items, 0..) |existing_path, i| {
            if (std.mem.eql(u8, existing_path, path)) {
                const removed_path = self.plugin_paths.swapRemove(i);
                self.allocator.free(removed_path);
                break;
            }
        }
    }

    /// Discover plugins in the search paths
    pub fn discoverPlugins(self: *PluginLoader) !std.ArrayList([]u8) {
        var discovered_plugins = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (discovered_plugins.items) |plugin_path| {
                self.allocator.free(plugin_path);
            }
            discovered_plugins.deinit();
        }

        for (self.plugin_paths.items) |search_path| {
            try self.discoverPluginsInPath(search_path, &discovered_plugins);
        }

        return discovered_plugins;
    }

    /// Discover plugins in a specific directory
    fn discoverPluginsInPath(self: *PluginLoader, path: []const u8, plugins: *std.ArrayList([]u8)) !void {
        var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return, // Path doesn't exist, skip
            error.NotDir => return, // Not a directory, skip
            else => return err,
        };
        defer dir.close();

        var iterator = dir.iterate();
        while (try iterator.next()) |entry| {
            if (entry.kind != .file) continue;

            const extension = getLibraryExtension();
            if (std.mem.endsWith(u8, entry.name, extension)) {
                const full_path = try std.fs.path.join(self.allocator, &[_][]const u8{ path, entry.name });
                errdefer self.allocator.free(full_path);
                try plugins.append(full_path);
            }
        }
    }

    /// Load a plugin from a file path
    pub fn loadPlugin(self: *PluginLoader, plugin_path: []const u8) !*const PluginInterface {
        // Check if already loaded
        for (self.loaded_libraries.items) |lib| {
            if (std.mem.eql(u8, lib.path, plugin_path)) {
                if (lib.interface) |existing_interface| {
                    return existing_interface;
                }
            }
        }

        // Load the library
        const handle = try self.loadLibrary(plugin_path);
        errdefer self.unloadLibrary(handle) catch {};

        // Get the factory function
        const factory_fn = try self.getSymbol(handle, PLUGIN_ENTRY_POINT, PluginFactoryFn);

        // Create the plugin interface
        const plugin_interface = factory_fn() orelse return PluginError.InvalidPlugin;

        // Validate the interface
        if (!plugin_interface.isValid()) {
            return PluginError.InvalidPlugin;
        }

        // Store the loaded library
        const owned_path = try self.allocator.dupe(u8, plugin_path);
        errdefer self.allocator.free(owned_path);

        try self.loaded_libraries.append(self.allocator, .{
            .path = owned_path,
            .handle = handle,
            .factory_fn = factory_fn,
            .interface = plugin_interface,
        });

        return plugin_interface;
    }

    /// Unload a plugin
    pub fn unloadPlugin(self: *PluginLoader, plugin_path: []const u8) !void {
        for (self.loaded_libraries.items, 0..) |*lib, i| {
            if (std.mem.eql(u8, lib.path, plugin_path)) {
                try self.unloadLibrary(lib.handle);
                self.allocator.free(lib.path);
                _ = self.loaded_libraries.swapRemove(i);
                return;
            }
        }
        return PluginError.PluginNotFound;
    }

    /// Get the list of loaded plugins
    pub fn getLoadedPlugins(self: *PluginLoader) []const LoadedLibrary {
        return self.loaded_libraries.items;
    }

    /// Platform-specific library loading
    fn loadLibrary(self: *PluginLoader, path: []const u8) !LibraryHandle {
        switch (builtin.os.tag) {
            .windows => {
                // Convert UTF-8 path to UTF-16 for Windows API
                const wide_path = try std.unicode.utf8ToUtf16LeAllocZ(self.allocator, path);
                defer self.allocator.free(wide_path);

                const handle = std.os.windows.kernel32.LoadLibraryW(wide_path.ptr);
                if (handle == null) {
                    return PluginError.LoadFailed;
                }
                return handle.?;
            },
            .linux, .macos, .freebsd, .openbsd, .netbsd, .dragonfly => {
                const c_path = try self.allocator.dupeZ(u8, path);
                defer self.allocator.free(c_path);

                const handle = std.c.dlopen(c_path, std.c.RTLD.LAZY);
                if (handle == null) {
                    return PluginError.LoadFailed;
                }
                return handle.?;
            },
            else => return PluginError.UnsupportedABI,
        }
    }

    /// Platform-specific library unloading
    fn unloadLibrary(self: *PluginLoader, handle: LibraryHandle) !void {
        _ = self;

        switch (builtin.os.tag) {
            .windows => {
                if (std.os.windows.kernel32.FreeLibrary(handle) == 0) {
                    return PluginError.LoadFailed;
                }
            },
            .linux, .macos, .freebsd, .openbsd, .netbsd, .dragonfly => {
                if (std.c.dlclose(handle) != 0) {
                    return PluginError.LoadFailed;
                }
            },
            else => return PluginError.UnsupportedABI,
        }
    }

    /// Platform-specific symbol resolution
    fn getSymbol(self: *PluginLoader, handle: LibraryHandle, symbol_name: []const u8, comptime T: type) !T {
        switch (builtin.os.tag) {
            .windows => {
                const symbol_name_z = try self.allocator.dupeZ(u8, symbol_name);
                defer self.allocator.free(symbol_name_z);

                const symbol = std.os.windows.kernel32.GetProcAddress(handle, symbol_name_z.ptr);
                if (symbol == null) {
                    return PluginError.SymbolNotFound;
                }
                return @ptrCast(symbol);
            },
            .linux, .macos, .freebsd, .openbsd, .netbsd, .dragonfly => {
                const symbol_name_z = try self.allocator.dupeZ(u8, symbol_name);
                defer self.allocator.free(symbol_name_z);

                const symbol = std.c.dlsym(handle, symbol_name_z);
                if (symbol == null) {
                    return PluginError.SymbolNotFound;
                }
                return @ptrCast(symbol);
            },
            else => return PluginError.UnsupportedABI,
        }
    }

    /// Get the platform-specific library extension
    fn getLibraryExtension() []const u8 {
        return switch (builtin.os.tag) {
            .windows => ".dll",
            .macos => ".dylib",
            .linux, .freebsd, .openbsd, .netbsd, .dragonfly => ".so",
            else => ".so", // Default to .so for Unix-like systems
        };
    }

    /// Get the platform-specific library prefix
    fn getLibraryPrefix() []const u8 {
        return switch (builtin.os.tag) {
            .windows => "",
            else => "lib",
        };
    }

    /// Construct a platform-specific library filename
    pub fn makeLibraryName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
        const prefix = getLibraryPrefix();
        const extension = getLibraryExtension();

        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ prefix, name, extension });
    }
};

/// Create a plugin loader instance
pub fn createLoader(allocator: std.mem.Allocator) PluginLoader {
    return PluginLoader.init(allocator);
}

// =============================================================================
// TESTS
// =============================================================================

test "PluginLoader initialization" {
    var loader = createLoader(std.testing.allocator);
    defer loader.deinit();

    try std.testing.expectEqual(@as(usize, 0), loader.loaded_libraries.items.len);
    try std.testing.expectEqual(@as(usize, 0), loader.plugin_paths.items.len);
}

test "Plugin path management" {
    var loader = createLoader(std.testing.allocator);
    defer loader.deinit();

    try loader.addPluginPath("/usr/local/lib/plugins");
    try loader.addPluginPath("/opt/plugins");

    try std.testing.expectEqual(@as(usize, 2), loader.plugin_paths.items.len);

    loader.removePluginPath("/usr/local/lib/plugins");
    try std.testing.expectEqual(@as(usize, 1), loader.plugin_paths.items.len);
}

test "Library name construction" {
    const allocator = std.testing.allocator;

    const lib_name = try PluginLoader.makeLibraryName(allocator, "myplugin");
    defer allocator.free(lib_name);

    switch (builtin.os.tag) {
        .windows => try std.testing.expectEqualStrings("myplugin.dll", lib_name),
        .macos => try std.testing.expectEqualStrings("libmyplugin.dylib", lib_name),
        .linux => try std.testing.expectEqualStrings("libmyplugin.so", lib_name),
        else => try std.testing.expectEqualStrings("libmyplugin.so", lib_name),
    }
}
