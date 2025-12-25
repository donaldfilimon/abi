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
