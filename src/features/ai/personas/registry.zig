//! Central registry for Multi-Persona AI Assistants.
//! Manages registration, discovery, and lifecycle of all AI personas.

const std = @import("std");
const types = @import("types.zig");
const config = @import("config.zig");

const sync = @import("../../../services/shared/sync.zig");
const Mutex = sync.Mutex;

/// Central registry managing the lifecycle and discovery of AI personas.
pub const PersonaRegistry = struct {
    allocator: std.mem.Allocator,
    /// Map of registered persona implementations.
    personas: std.AutoHashMapUnmanaged(types.PersonaType, types.PersonaInterface),
    /// Map of persona-specific configurations.
    configs: std.AutoHashMapUnmanaged(types.PersonaType, config.MultiPersonaConfig),
    /// Mutex for thread-safe access to the registry.
    mutex: Mutex,

    const Self = @This();

    /// Initialize a new persona registry.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .personas = .{},
            .configs = .{},
            .mutex = .{},
        };
    }

    /// Deinitialize the registry and free all resources.
    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.personas.deinit(self.allocator);
        self.configs.deinit(self.allocator);
    }

    /// Register a new persona implementation in the system.
    pub fn registerPersona(
        self: *Self,
        persona_type: types.PersonaType,
        persona: types.PersonaInterface,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.personas.put(self.allocator, persona_type, persona);
    }

    /// Set or update the configuration for a specific persona.
    pub fn configurePersona(
        self: *Self,
        persona_type: types.PersonaType,
        cfg: config.MultiPersonaConfig,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.configs.put(self.allocator, persona_type, cfg);
    }

    /// Retrieve a persona implementation by type.
    pub fn getPersona(self: *Self, persona_type: types.PersonaType) ?types.PersonaInterface {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.personas.get(persona_type);
    }

    /// Retrieve any registered persona type (useful for fallback selection).
    pub fn getAnyPersonaType(self: *Self) ?types.PersonaType {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.personas.keyIterator();
        if (it.next()) |key| {
            return key.*;
        }
        return null;
    }

    /// Retrieve the configuration for a specific persona.
    pub fn getConfiguration(self: *Self, persona_type: types.PersonaType) ?config.MultiPersonaConfig {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.configs.get(persona_type);
    }

    /// List all currently registered persona types.
    pub fn listRegisteredTypes(self: *Self, allocator: std.mem.Allocator) ![]types.PersonaType {
        self.mutex.lock();
        defer self.mutex.unlock();

        var list: std.ArrayListUnmanaged(types.PersonaType) = .{};
        errdefer list.deinit(allocator);

        var it = self.personas.keyIterator();
        while (it.next()) |key| {
            try list.append(allocator, key.*);
        }

        return list.toOwnedSlice(allocator);
    }

    /// Remove a persona from the registry.
    pub fn unregisterPersona(self: *Self, persona_type: types.PersonaType) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        const removed = self.personas.remove(persona_type);
        _ = self.configs.remove(persona_type);
        return removed;
    }
};

test {
    std.testing.refAllDecls(@This());
}
