//! Central registry for Multi-Profile AI Assistants.
//! Manages registration, discovery, and lifecycle of all AI profiles.

const std = @import("std");
const types = @import("types.zig");
const config = @import("config.zig");

const sync = @import("../../services/shared/mod.zig").sync;
const Mutex = sync.Mutex;

/// Central registry managing the lifecycle and discovery of AI profiles.
pub const ProfileRegistry = struct {
    allocator: std.mem.Allocator,
    /// Map of registered profile implementations.
    profiles: std.AutoHashMapUnmanaged(types.ProfileType, types.ProfileInterface),
    /// Map of profile-specific configurations.
    configs: std.AutoHashMapUnmanaged(types.ProfileType, config.MultiProfileConfig),
    /// Mutex for thread-safe access to the registry.
    mutex: Mutex,

    const Self = @This();

    /// Initialize a new profile registry.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .profiles = .empty,
            .configs = .empty,
            .mutex = .{},
        };
    }

    /// Deinitialize the registry and free all resources.
    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.profiles.deinit(self.allocator);
        self.configs.deinit(self.allocator);
    }

    /// Register a new profile implementation in the system.
    pub fn registerProfile(
        self: *Self,
        profile_type: types.ProfileType,
        profile: types.ProfileInterface,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.profiles.put(self.allocator, profile_type, profile);
    }

    /// Set or update the configuration for a specific profile.
    pub fn configureProfile(
        self: *Self,
        profile_type: types.ProfileType,
        cfg: config.MultiProfileConfig,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.configs.put(self.allocator, profile_type, cfg);
    }

    /// Retrieve a profile implementation by type.
    pub fn getProfile(self: *Self, profile_type: types.ProfileType) ?types.ProfileInterface {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.profiles.get(profile_type);
    }

    /// Retrieve any registered profile type (useful for fallback selection).
    pub fn getAnyProfileType(self: *Self) ?types.ProfileType {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.profiles.keyIterator();
        if (it.next()) |key| {
            return key.*;
        }
        return null;
    }

    /// Retrieve the configuration for a specific profile.
    pub fn getConfiguration(self: *Self, profile_type: types.ProfileType) ?config.MultiProfileConfig {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.configs.get(profile_type);
    }

    /// List all currently registered profile types.
    pub fn listRegisteredTypes(self: *Self, allocator: std.mem.Allocator) ![]types.ProfileType {
        self.mutex.lock();
        defer self.mutex.unlock();

        var list: std.ArrayListUnmanaged(types.ProfileType) = .empty;
        errdefer list.deinit(allocator);

        var it = self.profiles.keyIterator();
        while (it.next()) |key| {
            try list.append(allocator, key.*);
        }

        return list.toOwnedSlice(allocator);
    }

    /// Remove a profile from the registry.
    pub fn unregisterProfile(self: *Self, profile_type: types.ProfileType) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        const removed = self.profiles.remove(profile_type);
        _ = self.configs.remove(profile_type);
        return removed;
    }
};

test {
    std.testing.refAllDecls(@This());
}
