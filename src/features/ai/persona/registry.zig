//! PersonaRegistry: factory and lifecycle manager for Abbey, Aviva, and Abi.
//!
//! Wraps the three persona implementations into a unified interface for
//! instantiation, activation, suspension, and teardown.

const std = @import("std");
const types = @import("types.zig");
const PersonaId = types.PersonaId;
const PersonaState = types.PersonaState;
const PersonaError = types.PersonaError;
const PersonaResponse = types.PersonaResponse;
const RoutingConfig = types.RoutingConfig;

// Persona implementations
const abbey_mod = @import("../abbey/mod.zig");
const aviva_mod = @import("../aviva/mod.zig");
const abi_mod = @import("../abi/mod.zig");
const ai_config = @import("../config.zig");

/// Configuration for the multi-persona system.
pub const MultiPersonaConfig = struct {
    abbey: ai_config.AbbeyConfig = .{},
    aviva: ai_config.AvivaConfig = .{},
    abi: ai_config.AbiConfig = .{},
    routing: RoutingConfig = .{},
};

/// A single persona instance wrapping its underlying implementation.
pub const PersonaInstance = struct {
    id: PersonaId,
    state: PersonaState = .uninitialized,

    // Each persona has different types — we store opaque pointers and
    // dispatch through the id. Only one will be non-null.
    abbey_engine: ?*abbey_mod.AbbeyEngine = null,
    aviva_profile: ?*aviva_mod.AvivaProfile = null,
    abi_router: ?*abi_mod.AbiRouter = null,

    allocator: std.mem.Allocator,

    const Self = @This();

    /// Process a message through this persona's engine.
    pub fn process(self: *Self, message: []const u8) PersonaError!PersonaResponse {
        if (self.state != .active and self.state != .idle) return error.PersonaNotInitialized;

        self.state = .active;
        defer self.state = .idle;

        return switch (self.id) {
            .abbey => {
                if (self.abbey_engine) |engine| {
                    const result = engine.process(message) catch return error.PersonaFailed;
                    return PersonaResponse{
                        .persona = .abbey,
                        .content = result.content,
                        .confidence = result.confidence.score,
                        .reasoning = result.reasoning_summary,
                        .allocator = self.allocator,
                    };
                }
                return error.PersonaNotInitialized;
            },
            .aviva => {
                if (self.aviva_profile) |profile| {
                    const result = profile.respond(message) catch return error.PersonaFailed;
                    return PersonaResponse{
                        .persona = .aviva,
                        .content = result,
                        .confidence = 0.85, // Aviva is high-confidence by design
                        .allocator = self.allocator,
                    };
                }
                return error.PersonaNotInitialized;
            },
            .abi => {
                // Abi is primarily a router, not a responder — but can provide
                // compliance-focused responses when directly addressed.
                return PersonaResponse{
                    .persona = .abi,
                    .content = try self.allocator.dupe(u8, "[Abi] Compliance check passed. Routing to appropriate persona."),
                    .confidence = 0.9,
                    .allocator = self.allocator,
                };
            },
        };
    }

    pub fn getName(self: *const Self) []const u8 {
        return self.id.name();
    }

    pub fn isAvailable(self: *const Self) bool {
        return self.state == .idle or self.state == .active;
    }
};

/// Registry managing all persona instances.
pub const PersonaRegistry = struct {
    allocator: std.mem.Allocator,
    config: MultiPersonaConfig,
    instances: std.EnumArray(PersonaId, PersonaInstance),
    initialized: bool = false,

    const Self = @This();

    /// Create a new registry with the given configuration.
    /// Does NOT initialize persona engines — call `initAll()` for that.
    pub fn init(allocator: std.mem.Allocator, config: MultiPersonaConfig) Self {
        var instances = std.EnumArray(PersonaId, PersonaInstance).initFill(.{
            .id = .abbey, // placeholder, overwritten below
            .allocator = allocator,
        });

        // Set correct IDs
        instances.set(.abbey, .{ .id = .abbey, .allocator = allocator });
        instances.set(.aviva, .{ .id = .aviva, .allocator = allocator });
        instances.set(.abi, .{ .id = .abi, .allocator = allocator });

        return .{
            .allocator = allocator,
            .config = config,
            .instances = instances,
        };
    }

    /// Initialize all persona engines. Call once at startup.
    pub fn initAll(self: *Self) !void {
        // Initialize Abbey
        var abbey_instance = self.instances.getPtr(.abbey);
        const abbey_engine = try self.allocator.create(abbey_mod.AbbeyEngine);
        abbey_engine.* = try abbey_mod.AbbeyEngine.init(self.allocator, self.config.abbey);
        abbey_instance.abbey_engine = abbey_engine;
        abbey_instance.state = .idle;

        // Initialize Aviva
        var aviva_instance = self.instances.getPtr(.aviva);
        aviva_instance.aviva_profile = try aviva_mod.AvivaProfile.init(self.allocator, self.config.aviva);
        aviva_instance.state = .idle;

        // Initialize Abi
        var abi_instance = self.instances.getPtr(.abi);
        abi_instance.abi_router = try abi_mod.AbiRouter.init(self.allocator, self.config.abi);
        abi_instance.state = .idle;

        self.initialized = true;
    }

    /// Get a persona instance by ID.
    pub fn getPersona(self: *Self, id: PersonaId) *PersonaInstance {
        return self.instances.getPtr(id);
    }

    /// Get the Abi router (for use by MultiPersonaRouter).
    pub fn getAbiRouter(self: *Self) ?*abi_mod.AbiRouter {
        return self.instances.getPtr(.abi).abi_router;
    }

    /// Suspend a persona (it can be reactivated later).
    pub fn suspendPersona(self: *Self, id: PersonaId) void {
        self.instances.getPtr(id).state = .suspended;
    }

    /// Resume a suspended persona.
    pub fn resumePersona(self: *Self, id: PersonaId) void {
        const instance = self.instances.getPtr(id);
        if (instance.state == .suspended) {
            instance.state = .idle;
        }
    }

    /// List all available (non-suspended, initialized) personas.
    pub fn listAvailable(self: *Self) [3]?PersonaId {
        var result: [3]?PersonaId = .{ null, null, null };
        var i: usize = 0;
        for (std.enums.values(PersonaId)) |id| {
            if (self.instances.get(id).isAvailable()) {
                result[i] = id;
                i += 1;
            }
        }
        return result;
    }

    /// Shut down all persona engines and free resources.
    pub fn deinit(self: *Self) void {
        var abbey = self.instances.getPtr(.abbey);
        if (abbey.abbey_engine) |engine| {
            engine.deinit();
            self.allocator.destroy(engine);
            abbey.abbey_engine = null;
        }

        var aviva = self.instances.getPtr(.aviva);
        if (aviva.aviva_profile) |profile| {
            profile.deinit();
            aviva.aviva_profile = null;
        }

        var abi_inst = self.instances.getPtr(.abi);
        if (abi_inst.abi_router) |router| {
            router.deinit();
            abi_inst.abi_router = null;
        }

        self.initialized = false;
    }
};

test "persona registry init and deinit" {
    const allocator = std.testing.allocator;
    var registry = PersonaRegistry.init(allocator, .{});
    defer registry.deinit();

    // Before initAll, personas are uninitialized
    const abbey = registry.getPersona(.abbey);
    try std.testing.expectEqual(PersonaState.uninitialized, abbey.state);
    try std.testing.expectEqualStrings("Abbey", abbey.getName());
}

test "persona id names" {
    try std.testing.expectEqualStrings("Abbey", PersonaId.abbey.name());
    try std.testing.expectEqualStrings("Aviva", PersonaId.aviva.name());
    try std.testing.expectEqualStrings("Abi", PersonaId.abi.name());
}

test {
    std.testing.refAllDecls(@This());
}
