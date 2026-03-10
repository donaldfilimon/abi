//! Orchestrator: multi-backend inference and skill sync.

const std = @import("std");
const types = @import("protocol/types");
const backend = @import("backends/backend");
const skill_registry = @import("skills/registry");
const sync_manager = @import("sync/manager");

pub const Orchestrator = struct {
    allocator: std.mem.Allocator,
    backends: std.ArrayListUnmanaged(BackendEntry),
    skills: skill_registry.SkillRegistry,
    sync: sync_manager.SyncManager,

    const BackendEntry = struct {
        config: types.BackendConfig,
        iface: backend.BackendInterface,
    };

    pub fn init(allocator: std.mem.Allocator) Orchestrator {
        const self = Orchestrator{
            .allocator = allocator,
            .backends = .{},
            .skills = skill_registry.SkillRegistry.init(allocator),
            .sync = sync_manager.SyncManager.init(allocator),
        };
        return self;
    }

    pub fn deinit(self: *Orchestrator) void {
        self.backends.deinit(self.allocator);
        self.skills.deinit();
        self.sync.deinit();
    }

    pub fn addBackend(self: *Orchestrator, config: types.BackendConfig, iface: backend.BackendInterface) !void {
        try self.backends.append(self.allocator, .{ .config = config, .iface = iface });
    }

    pub fn infer(self: *Orchestrator, request: types.InferenceRequest) !types.InferenceResponse {
        if (self.backends.items.len == 0) return error.NoBackendRegistered;
        const entry = &self.backends.items[0];
        return entry.iface.run(self.allocator, request);
    }
};
