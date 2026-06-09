const std = @import("std");
const foundation_pool = @import("../../foundation/pool_allocator.zig");

pub const Point3D = struct {
    x: f32,
    y: f32,
    z: f32,
};

pub const SpatialRecord3D = struct {
    id: u32,
    point: Point3D,
    payload: []const u8,
    pooled_block: ?[]u8 = null,
};

pub const DistanceMetric = enum {
    euclidean,
    manhattan,
    cosine,
};

pub const SpatialSearchResult = struct {
    id: u32,
    distance: f32,
    point: Point3D,
    payload: []const u8,
};

const PayloadAllocation = struct {
    payload: []const u8,
    pooled_block: ?[]u8 = null,
};

pub fn euclideanDistance(p1: Point3D, p2: Point3D) f32 {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;
    return @sqrt(dx * dx + dy * dy + dz * dz);
}

pub fn manhattanDistance(p1: Point3D, p2: Point3D) f32 {
    return @abs(p1.x - p2.x) + @abs(p1.y - p2.y) + @abs(p1.z - p2.z);
}

pub fn cosineDistance(p1: Point3D, p2: Point3D) f32 {
    const dot = p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
    const norm1 = @sqrt(p1.x * p1.x + p1.y * p1.y + p1.z * p1.z);
    const norm2 = @sqrt(p2.x * p2.x + p2.y * p2.y + p2.z * p2.z);
    if (norm1 == 0 or norm2 == 0) return 1.0;
    const similarity = dot / (norm1 * norm2);
    return 1.0 - similarity;
}

pub fn calculateDistance(p1: Point3D, p2: Point3D, metric: DistanceMetric) f32 {
    return switch (metric) {
        .euclidean => euclideanDistance(p1, p2),
        .manhattan => manhattanDistance(p1, p2),
        .cosine => cosineDistance(p1, p2),
    };
}

pub const SpatialIndex3D = struct {
    allocator: std.mem.Allocator,
    pool_alloc: ?*foundation_pool.PoolAllocator = null,
    records: std.ArrayListUnmanaged(SpatialRecord3D) = .empty,

    pub fn init(allocator: std.mem.Allocator) SpatialIndex3D {
        return initWithPool(allocator, null);
    }

    pub fn initWithPool(allocator: std.mem.Allocator, pool_alloc: ?*foundation_pool.PoolAllocator) SpatialIndex3D {
        return .{
            .allocator = allocator,
            .pool_alloc = pool_alloc,
        };
    }

    pub fn deinit(self: *SpatialIndex3D) void {
        for (self.records.items) |rec| {
            self.freePayload(rec);
        }
        self.records.deinit(self.allocator);
    }

    pub fn insert(self: *SpatialIndex3D, id: u32, point: Point3D, payload: []const u8) !void {
        const owned = try self.dupePayload(payload);
        errdefer self.freePayloadAllocation(owned);

        try self.records.append(self.allocator, .{
            .id = id,
            .point = point,
            .payload = owned.payload,
            .pooled_block = owned.pooled_block,
        });
    }

    fn dupePayload(self: *SpatialIndex3D, payload: []const u8) !PayloadAllocation {
        if (self.pool_alloc) |pool| {
            if (payload.len <= pool.block_size) {
                const block = try pool.alloc();
                @memcpy(block[0..payload.len], payload);
                return .{
                    .payload = block[0..payload.len],
                    .pooled_block = block,
                };
            }
        }

        const owned_payload = try self.allocator.dupe(u8, payload);
        return .{
            .payload = owned_payload,
            .pooled_block = null,
        };
    }

    fn freePayload(self: *SpatialIndex3D, rec: SpatialRecord3D) void {
        self.freePayloadAllocation(.{ .payload = rec.payload, .pooled_block = rec.pooled_block });
    }

    fn freePayloadAllocation(self: *SpatialIndex3D, allocation: PayloadAllocation) void {
        if (allocation.pooled_block) |block| {
            if (self.pool_alloc) |pool| {
                pool.free(block);
                return;
            }
        }
        self.allocator.free(allocation.payload);
    }

    pub fn count(self: *const SpatialIndex3D) usize {
        return self.records.items.len;
    }

    pub fn radiusSearch(self: *const SpatialIndex3D, center: Point3D, radius: f32, metric: DistanceMetric) ![]SpatialSearchResult {
        var results: std.ArrayListUnmanaged(SpatialSearchResult) = .empty;
        errdefer results.deinit(self.allocator);

        for (self.records.items) |rec| {
            const dist = calculateDistance(center, rec.point, metric);
            if (dist <= radius) {
                try results.append(self.allocator, .{
                    .id = rec.id,
                    .distance = dist,
                    .point = rec.point,
                    .payload = rec.payload,
                });
            }
        }

        // Sort by distance (closest first)
        std.mem.sort(SpatialSearchResult, results.items, {}, sortSearchResult);
        return try results.toOwnedSlice(self.allocator);
    }

    pub fn nearestNeighbors(self: *const SpatialIndex3D, center: Point3D, k: usize, metric: DistanceMetric) ![]SpatialSearchResult {
        var results: std.ArrayListUnmanaged(SpatialSearchResult) = .empty;
        errdefer results.deinit(self.allocator);

        for (self.records.items) |rec| {
            const dist = calculateDistance(center, rec.point, metric);
            try results.append(self.allocator, .{
                .id = rec.id,
                .distance = dist,
                .point = rec.point,
                .payload = rec.payload,
            });
        }

        std.mem.sort(SpatialSearchResult, results.items, {}, sortSearchResult);

        const return_count = @min(k, results.items.len);
        const owned_slice = try self.allocator.alloc(SpatialSearchResult, return_count);
        @memcpy(owned_slice, results.items[0..return_count]);
        results.deinit(self.allocator);
        return owned_slice;
    }
};

fn sortSearchResult(_: void, lhs: SpatialSearchResult, rhs: SpatialSearchResult) bool {
    return lhs.distance < rhs.distance;
}

test "SpatialIndex3D insert and searches" {
    var index = SpatialIndex3D.init(std.testing.allocator);
    defer index.deinit();

    try index.insert(1, .{ .x = 0.0, .y = 0.0, .z = 0.0 }, "Origin");
    try index.insert(2, .{ .x = 1.0, .y = 0.0, .z = 0.0 }, "Unit X");
    try index.insert(3, .{ .x = 0.0, .y = 2.0, .z = 0.0 }, "Y Coordinate 2");
    try index.insert(4, .{ .x = 1.0, .y = 1.0, .z = 1.0 }, "Diagonal Node");

    try std.testing.expectEqual(@as(usize, 4), index.count());

    // 1. Radius Search (radius = 1.5, Euclidean)
    const radius_results = try index.radiusSearch(.{ .x = 0.0, .y = 0.0, .z = 0.0 }, 1.5, .euclidean);
    defer std.testing.allocator.free(radius_results);

    try std.testing.expectEqual(@as(usize, 2), radius_results.len); // Should match Origin and Unit X
    try std.testing.expectEqual(@as(u32, 1), radius_results[0].id);
    try std.testing.expectEqual(@as(u32, 2), radius_results[1].id);

    // 2. Nearest Neighbors Search (K = 2, Euclidean)
    const nn_results = try index.nearestNeighbors(.{ .x = 0.0, .y = 0.0, .z = 0.0 }, 2, .euclidean);
    defer std.testing.allocator.free(nn_results);

    try std.testing.expectEqual(@as(usize, 2), nn_results.len);
    try std.testing.expectEqual(@as(u32, 1), nn_results[0].id);
    try std.testing.expectEqual(@as(u32, 2), nn_results[1].id);
}

test "SpatialIndex3D distance metrics" {
    const p1 = Point3D{ .x = 1.0, .y = 2.0, .z = 3.0 };
    const p2 = Point3D{ .x = 4.0, .y = 6.0, .z = 8.0 };

    try std.testing.expectApproxEqAbs(@as(f32, 7.0710678), euclideanDistance(p1, p2), 0.0001);
    try std.testing.expectEqual(@as(f32, 12.0), manhattanDistance(p1, p2));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0074167), cosineDistance(p1, p2), 0.0001);
}

test {
    std.testing.refAllDecls(@This());
}
