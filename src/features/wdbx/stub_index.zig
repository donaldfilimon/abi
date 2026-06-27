const std = @import("std");
const memory = @import("../../core/memory.zig");
const types = @import("stub_types.zig");

pub const MAX_LAYERS = types.MAX_LAYERS;
pub const M = 16;
pub const EF_CONSTRUCTION = 40;
pub const EF_SEARCH = 32;

pub const HnswNode = struct {
    pub fn initEdges(allocator: std.mem.Allocator) [MAX_LAYERS]std.ArrayListUnmanaged(u32) {
        _ = allocator;
        var arr: [MAX_LAYERS]std.ArrayListUnmanaged(u32) = undefined;
        var i: usize = 0;
        while (i < MAX_LAYERS) : (i += 1) {
            arr[i] = .empty;
        }
        return arr;
    }

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub const VectorStorage = struct {
    pub fn init(allocator: std.mem.Allocator, dimensions: usize, initial_capacity: usize) VectorStorage {
        _ = allocator;
        _ = dimensions;
        _ = initial_capacity;
        return .{};
    }

    pub fn deinit(self: *VectorStorage) void {
        _ = self;
    }

    pub fn setTracker(self: *VectorStorage, tracker: *memory.MemoryTracker) void {
        _ = self;
        _ = tracker;
    }

    pub fn insert(self: *VectorStorage, id: u32, values: []const f32) !void {
        _ = self;
        _ = id;
        _ = values;
        return error.FeatureDisabled;
    }

    pub fn get(self: *const VectorStorage, id: u32) ?[]const f32 {
        _ = self;
        _ = id;
        return null;
    }

    pub fn contains(self: *const VectorStorage, id: u32) bool {
        _ = self;
        _ = id;
        return false;
    }
};

pub const Candidate = struct {
    id: u32,
    distance: f32,
};

pub fn cosineDistanceSIMD(a: []const f32, b: []const f32) f32 {
    _ = a;
    _ = b;
    return 1.0;
}

pub fn HnswIndex(comptime D: usize) type {
    return struct {
        storage: VectorStorage,

        pub fn init(allocator: std.mem.Allocator) @This() {
            return .{ .storage = VectorStorage.init(allocator, D, 64) };
        }

        pub fn deinit(self: *@This()) void {
            self.storage.deinit();
        }

        pub fn setTracker(self: *@This(), tracker: *memory.MemoryTracker) void {
            _ = self;
            _ = tracker;
        }

        pub fn insert(self: *@This(), id: u32, values: []const f32) !void {
            _ = self;
            _ = id;
            _ = values;
            return error.FeatureDisabled;
        }

        pub fn search(self: *@This(), query: []const f32, limit: usize) ![]types.SearchResult {
            _ = self;
            _ = query;
            _ = limit;
            return error.FeatureDisabled;
        }

        pub fn count(self: *const @This()) usize {
            _ = self;
            return 0;
        }
    };
}
