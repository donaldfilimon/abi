//! Fault injection helper for Raft testing.

const std = @import("std");

/// Fault injection helper for testing network partitions.
/// Tracks blocked communication routes between node pairs.
pub const FaultInjector = struct {
    /// Blocked routes stored as "src_id->dst_id" keys.
    blocked_routes: std.StringHashMapUnmanaged(void),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) FaultInjector {
        return FaultInjector{
            .blocked_routes = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FaultInjector) void {
        var iter = self.blocked_routes.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.blocked_routes.deinit(self.allocator);
    }

    /// Block communication in both directions between two nodes.
    pub fn simulatePartition(self: *FaultInjector, node_a: []const u8, node_b: []const u8) !void {
        try self.blockRoute(node_a, node_b);
        try self.blockRoute(node_b, node_a);
    }

    /// Restore communication in both directions between two nodes.
    pub fn simulateHeal(self: *FaultInjector, node_a: []const u8, node_b: []const u8) void {
        self.unblockRoute(node_a, node_b);
        self.unblockRoute(node_b, node_a);
    }

    /// Check if communication from src to dst is blocked.
    pub fn isBlocked(self: *const FaultInjector, src: []const u8, dst: []const u8) bool {
        // Build route key on stack for lookup
        var buf: [512]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}->{s}", .{ src, dst }) catch return false;
        return self.blocked_routes.contains(key);
    }

    fn blockRoute(self: *FaultInjector, src: []const u8, dst: []const u8) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}->{s}", .{ src, dst });
        errdefer self.allocator.free(key);
        try self.blocked_routes.put(self.allocator, key, {});
    }

    fn unblockRoute(self: *FaultInjector, src: []const u8, dst: []const u8) void {
        var buf: [512]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}->{s}", .{ src, dst }) catch return;
        if (self.blocked_routes.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
        }
    }
};
