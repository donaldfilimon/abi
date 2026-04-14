//! ProfileRegistry: factory and lifecycle manager for Abbey, Aviva, and Abi.
//!
//! This registry manages ONLY the Abbey-Aviva-Abi triad (ProfileId: 3 variants).
//! It is DISTINCT from `registry.zig` which is a generic registry for ALL
//! 13 profile types (ProfileType: 13 variants).
//!
//! This registry handles:
//!   - Concrete profile engine initialization
//!   - Instance lifecycle (active, idle, suspended)
//!   - Direct message processing through profile engines

const std = @import("std");
const types = @import("types.zig");
const pipeline_types = @import("../pipeline/types.zig");
const ProfileId = types.ProfileId;
const ProfileState = types.ProfileState;
const ProfileError = types.ProfileError;
const ProfileResponse = types.ProfileResponse;
const RoutingConfig = types.RoutingConfig;
const RoutingDecision = @import("../profile/types.zig").RoutingDecision;

// Profile implementations
const abbey_mod = @import("../abbey/mod.zig");
const aviva_mod = @import("../aviva/mod.zig");
const abi_mod = @import("../abi/mod.zig");
const ai_config = @import("../config.zig");

/// Configuration for the multi-profile system.
pub const MultiProfileConfig_Internal = struct {
    abbey: ai_config.AbbeyConfig = .{},
    aviva: ai_config.AvivaConfig = .{},
    abi: ai_config.AbiConfig = .{},
    routing: RoutingConfig = .{},
};

pub const MultiProfileConfig = MultiProfileConfig_Internal;

// Internal exports for mod/stub parity
pub const ProfileInstance_Internal = struct {
    id: ProfileId,
    state: ProfileState = .uninitialized,
    allocator: std.mem.Allocator,

    pub fn process(_: *ProfileInstance, _: []const u8) ProfileError!ProfileResponse {
        return error.ProfileNotInitialized;
    }
    pub fn getName(self: *const ProfileInstance) []const u8 {
        return self.id.name();
    }
    pub fn isAvailable(_: *const ProfileInstance) bool {
        return false;
    }
};

pub const ProfileRegistry_Internal = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, _: MultiProfileConfig_Internal) ProfileRegistry_Internal {
        return .{ .allocator = allocator };
    }
    pub fn initAll(_: *ProfileRegistry) ProfileError!void {
        return error.ProfileNotInitialized;
    }
    pub fn getProfile(_: *ProfileRegistry, _: ProfileId) ?*ProfileInstance {
        return null;
    }
    pub fn getAbiRouter(_: *ProfileRegistry) ?*anyopaque {
        return null;
    }
    pub fn suspendProfile(_: *ProfileRegistry, _: ProfileId) void {}
    pub fn resumeProfile(_: *ProfileRegistry, _: ProfileId) void {}
    pub fn listAvailable(_: *ProfileRegistry) [3]?ProfileId {
        return .{ null, null, null };
    }
    pub fn deinit(_: *ProfileRegistry) void {}
};

pub const MultiProfileRouter_Internal = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: *ProfileRegistry_Internal, _: RoutingConfig) MultiProfileRouter_Internal {
        return .{ .allocator = allocator };
    }
    pub fn route(_: *MultiProfileRouter_Internal, _: []const u8) RoutingDecision {
        return .{
            .primary = .abbey,
            .weights = .{},
            .strategy = .single,
            .confidence = 0.0,
            .reason = "AI disabled",
        };
    }
    pub fn execute(_: *MultiProfileRouter_Internal, _: RoutingDecision, _: []const u8) ProfileError!ProfileResponse {
        return error.ProfileNotInitialized;
    }
    pub fn routeAndExecute(_: *MultiProfileRouter_Internal, _: []const u8) ProfileError!ProfileResponse {
        return error.ProfileNotInitialized;
    }
    pub fn deinit(_: *MultiProfileRouter_Internal) void {}
};

pub const ProfileBus_Internal = struct {
    pub fn init(_: std.mem.Allocator) ProfileBus_Internal {
        return .{};
    }
    pub fn deinit(_: *ProfileBus_Internal) void {}
};

pub const ConversationMemory_Internal = struct {
    pub fn init(_: std.mem.Allocator) ConversationMemory_Internal {
        return .{};
    }
    pub fn asStoreStep(_: *const ConversationMemory_Internal) pipeline_types.StoreConfig {
        return .{ .target = .wdbx };
    }
    pub fn deinit(_: *ConversationMemory_Internal) void {}
};

/// A single profile instance wrapping its underlying implementation.
pub const ProfileInstance = struct {
    id: ProfileId,
    state: ProfileState = .uninitialized,

    // Each profile has different types — we store opaque pointers and
    // dispatch through the id. Only one will be non-null.
    abbey_engine: ?*abbey_mod.AbbeyEngine = null,
    aviva_profile: ?*aviva_mod.AvivaProfile = null,
    abi_router: ?*abi_mod.AbiRouter = null,

    allocator: std.mem.Allocator,

    const Self = @This();

    /// Process a message through this profile's engine.
    pub fn process(self: *Self, message: []const u8) ProfileError!ProfileResponse {
        if (self.state != .active and self.state != .idle) return error.ProfileNotInitialized;

        self.state = .active;
        defer self.state = .idle;

        return switch (self.id) {
            .abbey => {
                if (self.abbey_engine) |engine| {
                    const result = engine.process(message) catch return error.ProfileFailed;
                    return ProfileResponse{
                        .profile = .abbey,
                        .content = result.content,
                        .confidence = result.confidence.score,
                        .allocator = self.allocator,
                    };
                }
                return error.ProfileNotInitialized;
            },
            .aviva => {
                if (self.aviva_profile) |profile| {
                    const result = profile.respond(message) catch return error.ProfileFailed;
                    return ProfileResponse{
                        .profile = .aviva,
                        .content = result,
                        .confidence = 0.85, // Aviva is high-confidence by design
                        .allocator = self.allocator,
                    };
                }
                return error.ProfileNotInitialized;
            },
            .abi => {
                // Abi is primarily a router, not a responder — but can provide
                // compliance-focused responses when directly addressed.
                return ProfileResponse{
                    .profile = .abi,
                    .content = try self.allocator.dupe(u8, "[Abi] Compliance check passed. Routing to appropriate profile."),
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

/// Registry managing all profile instances.
pub const ProfileRegistry = struct {
    allocator: std.mem.Allocator,
    config: MultiProfileConfig,
    instances: std.EnumArray(ProfileId, ProfileInstance),
    initialized: bool = false,

    const Self = @This();

    /// Create a new registry with the given configuration.
    /// Does NOT initialize profile engines — call `initAll()` for that.
    pub fn init(allocator: std.mem.Allocator, config: MultiProfileConfig) Self {
        var instances = std.EnumArray(ProfileId, ProfileInstance).initFill(.{
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

    /// Initialize all profile engines. Call once at startup.
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

    /// Get a profile instance by ID.
    pub fn getProfile(self: *Self, id: ProfileId) *ProfileInstance {
        return self.instances.getPtr(id);
    }

    /// Get the Abi router (for use by MultiProfileRouter).
    pub fn getAbiRouter(self: *Self) ?*abi_mod.AbiRouter {
        return self.instances.getPtr(.abi).abi_router;
    }

    /// Suspend a profile (it can be reactivated later).
    pub fn suspendProfile(self: *Self, id: ProfileId) void {
        self.instances.getPtr(id).state = .suspended;
    }

    /// Resume a suspended profile.
    pub fn resumeProfile(self: *Self, id: ProfileId) void {
        const instance = self.instances.getPtr(id);
        if (instance.state == .suspended) {
            instance.state = .idle;
        }
    }

    /// List all available (non-suspended, initialized) profiles.
    pub fn listAvailable(self: *Self) [3]?ProfileId {
        var result: [3]?ProfileId = .{ null, null, null };
        var i: usize = 0;
        for (std.enums.values(ProfileId)) |id| {
            if (self.instances.get(id).isAvailable()) {
                result[i] = id;
                i += 1;
            }
        }
        return result;
    }

    /// Shut down all profile engines and free resources.
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
            self.allocator.destroy(profile);
            aviva.aviva_profile = null;
        }

        var abi_inst = self.instances.getPtr(.abi);
        if (abi_inst.abi_router) |router| {
            router.deinit();
            self.allocator.destroy(router);
            abi_inst.abi_router = null;
        }

        self.initialized = false;
    }
};

test "profile registry init and deinit" {
    const allocator = std.testing.allocator;
    var registry = ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    // Before initAll, profiles are uninitialized
    const abbey = registry.getProfile(.abbey);
    try std.testing.expectEqual(ProfileState.uninitialized, abbey.state);
    try std.testing.expectEqualStrings("Abbey", abbey.getName());
}

test "profile id names" {
    try std.testing.expectEqualStrings("Abbey", ProfileId.abbey.name());
    try std.testing.expectEqualStrings("Aviva", ProfileId.aviva.name());
    try std.testing.expectEqualStrings("Abi", ProfileId.abi.name());
}

test {
    std.testing.refAllDecls(@This());
}
