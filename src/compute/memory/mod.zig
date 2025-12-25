const std = @import("std");

pub const StableAllocator = struct {
    gpa: std.heap.GeneralPurposeAllocator(.{}) = .{},

    pub fn allocator(self: *StableAllocator) std.mem.Allocator {
        return self.gpa.allocator();
    }

    pub fn deinit(self: *StableAllocator) void {
        _ = self.gpa.deinit();
    }
};

pub const WorkerArena = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) WorkerArena {
        return .{ .arena = std.heap.ArenaAllocator.init(backing_allocator) };
    }

    pub fn allocator(self: *WorkerArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn reset(self: *WorkerArena) void {
        self.arena.reset(.retain_capacity);
    }

    pub fn deinit(self: *WorkerArena) void {
        self.arena.deinit();
    }
};

test "stable allocator allocates" {
    var stable = StableAllocator{};
    defer stable.deinit();
    const allocator = stable.allocator();
    const buffer = try allocator.alloc(u8, 16);
    allocator.free(buffer);
}

test "worker arena reset" {
    var arena = WorkerArena.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const first = try allocator.alloc(u8, 8);
    _ = first;
    arena.reset();
    const second = try allocator.alloc(u8, 8);
    try std.testing.expectEqual(@as(usize, 8), second.len);
}
