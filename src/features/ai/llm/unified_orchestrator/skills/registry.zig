//! Skill registry stub (WDBX-backed skill sync).

const std = @import("std");
const types = @import("../protocol/types.zig");

pub const SkillRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SkillRegistry {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *SkillRegistry) void {
        _ = self;
    }

    pub fn register(self: *SkillRegistry, skill: types.Skill) !void {
        _ = self;
        _ = skill;
    }

    pub fn get(self: *const SkillRegistry, id: []const u8) ?types.Skill {
        _ = self;
        _ = id;
        return null;
    }
};
