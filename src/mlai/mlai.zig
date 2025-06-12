const std = @import("std");
const agent = @import("../agent.zig");
const wdbx = @import("mlai/wdbx/db.zig");

pub const MLAISystem = struct {
    allocator: std.mem.Allocator,
    db: wdbx.Database,

    pub fn init(alloc: std.mem.Allocator, cfg: wdbx.Config) !MLAISystem {
        return MLAISystem{
            .allocator = alloc,
            .db = try wdbx.Database.init(alloc, cfg),
        };
    }

    pub fn deinit(self: *MLAISystem) void {
        self.db.deinit();
    }

    pub fn processRequest(self: *MLAISystem, query: []const u8) ![]u8 {
        const persona = router(query);
        var response = try agent.respond(persona, query, self.allocator);
        errdefer self.allocator.free(response);
        try self.db.storeInteraction(query, response, persona);
        return response;
    }
};

pub fn router(query: []const u8) agent.PersonaType {
    if (std.mem.indexOf(u8, query, "help") != null) return .EmpatheticAnalyst;
    if (std.mem.indexOf(u8, query, "explain") != null) return .DirectExpert;
    return .AdaptiveModerator;
}

pub test "process request stores data" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var system = try MLAISystem.init(gpa.allocator(), .{ .shard_count = 2 });
    defer system.deinit();
    const reply = try system.processRequest("help me");
    defer gpa.allocator().free(reply);
    const e = system.db.retrieve("help me") orelse return error.NotFound;
    try std.testing.expectEqualStrings(e.value, reply);
}
