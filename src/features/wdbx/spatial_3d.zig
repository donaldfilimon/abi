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

const OCTREE_MIN_POINTS: usize = 64;
const OCTREE_LEAF_CAPACITY: usize = 8;
const OCTREE_MAX_DEPTH: u32 = 16;
const OCTREE_NULL: u32 = std.math.maxInt(u32);

const BoundingBox = struct {
    min: Point3D,
    max: Point3D,

    fn center(self: BoundingBox) Point3D {
        return .{
            .x = (self.min.x + self.max.x) * 0.5,
            .y = (self.min.y + self.max.y) * 0.5,
            .z = (self.min.z + self.max.z) * 0.5,
        };
    }

    /// Minimum possible distance from `p` to any point inside this box under
    /// `metric`. Used to prune subtrees: if this exceeds the query radius or
    /// the current k-th best distance, no point inside the box can qualify.
    /// Only valid for `.euclidean` and `.manhattan` -- callers must route
    /// `.cosine` to the linear-scan path before reaching this function.
    fn minDistanceTo(self: BoundingBox, p: Point3D, metric: DistanceMetric) f32 {
        const dx = @max(0.0, @max(self.min.x - p.x, p.x - self.max.x));
        const dy = @max(0.0, @max(self.min.y - p.y, p.y - self.max.y));
        const dz = @max(0.0, @max(self.min.z - p.z, p.z - self.max.z));
        return switch (metric) {
            .euclidean => @sqrt(dx * dx + dy * dy + dz * dz),
            .manhattan => dx + dy + dz,
            .cosine => unreachable,
        };
    }
};

fn computeBounds(records: []const SpatialRecord3D) BoundingBox {
    var min = records[0].point;
    var max = records[0].point;
    for (records[1..]) |rec| {
        min.x = @min(min.x, rec.point.x);
        min.y = @min(min.y, rec.point.y);
        min.z = @min(min.z, rec.point.z);
        max.x = @max(max.x, rec.point.x);
        max.y = @max(max.y, rec.point.y);
        max.z = @max(max.z, rec.point.z);
    }
    const eps: f32 = 1e-3;
    return .{
        .min = .{ .x = min.x - eps, .y = min.y - eps, .z = min.z - eps },
        .max = .{ .x = max.x + eps, .y = max.y + eps, .z = max.z + eps },
    };
}

fn octantOf(p: Point3D, center: Point3D) usize {
    var oct: usize = 0;
    if (p.x >= center.x) oct |= 1;
    if (p.y >= center.y) oct |= 2;
    if (p.z >= center.z) oct |= 4;
    return oct;
}

fn childBounds(bounds: BoundingBox, center: Point3D, octant: usize) BoundingBox {
    const x_hi = (octant & 1) != 0;
    const y_hi = (octant & 2) != 0;
    const z_hi = (octant & 4) != 0;
    return .{
        .min = .{
            .x = if (x_hi) center.x else bounds.min.x,
            .y = if (y_hi) center.y else bounds.min.y,
            .z = if (z_hi) center.z else bounds.min.z,
        },
        .max = .{
            .x = if (x_hi) bounds.max.x else center.x,
            .y = if (y_hi) bounds.max.y else center.y,
            .z = if (z_hi) bounds.max.z else center.z,
        },
    };
}

const OctreeNode = struct {
    bounds: BoundingBox,
    children: ?[8]u32 = null,
    point_indices: std.ArrayListUnmanaged(u32) = .empty,
};

const Octree = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayListUnmanaged(OctreeNode) = .empty,
    root: u32 = OCTREE_NULL,

    fn deinit(self: *Octree) void {
        for (self.nodes.items) |*node| node.point_indices.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    fn build(allocator: std.mem.Allocator, records: []const SpatialRecord3D) !Octree {
        var tree = Octree{ .allocator = allocator };
        errdefer tree.deinit();
        if (records.len == 0) return tree;

        const bounds = computeBounds(records);
        const indices = try allocator.alloc(u32, records.len);
        defer allocator.free(indices);
        for (indices, 0..) |*v, i| v.* = @intCast(i);

        tree.root = try tree.buildNode(records, indices, bounds, 0);
        return tree;
    }

    fn buildNode(self: *Octree, records: []const SpatialRecord3D, indices: []const u32, bounds: BoundingBox, depth: u32) !u32 {
        const node_idx: u32 = @intCast(self.nodes.items.len);
        try self.nodes.append(self.allocator, .{ .bounds = bounds });

        if (indices.len <= OCTREE_LEAF_CAPACITY or depth >= OCTREE_MAX_DEPTH) {
            var leaf_points: std.ArrayListUnmanaged(u32) = .empty;
            try leaf_points.appendSlice(self.allocator, indices);
            self.nodes.items[node_idx].point_indices = leaf_points;
            return node_idx;
        }

        const center = bounds.center();
        var buckets: [8]std.ArrayListUnmanaged(u32) = undefined;
        for (&buckets) |*bucket| bucket.* = .empty;
        defer for (&buckets) |*bucket| bucket.deinit(self.allocator);

        for (indices) |i| {
            const octant = octantOf(records[i].point, center);
            try buckets[octant].append(self.allocator, i);
        }

        var children: [8]u32 = undefined;
        for (0..8) |octant| {
            const bounds_for_child = childBounds(bounds, center, octant);
            children[octant] = try self.buildNode(records, buckets[octant].items, bounds_for_child, depth + 1);
        }
        self.nodes.items[node_idx].children = children;
        return node_idx;
    }
};

fn octreeRadiusWalk(
    tree: *const Octree,
    records: []const SpatialRecord3D,
    node_idx: u32,
    center: Point3D,
    radius: f32,
    metric: DistanceMetric,
    results: *std.ArrayListUnmanaged(SpatialSearchResult),
    allocator: std.mem.Allocator,
) !void {
    const node = &tree.nodes.items[node_idx];
    if (node.bounds.minDistanceTo(center, metric) > radius) return;

    if (node.children) |children| {
        for (children) |child_idx| {
            try octreeRadiusWalk(tree, records, child_idx, center, radius, metric, results, allocator);
        }
        return;
    }

    for (node.point_indices.items) |i| {
        const rec = records[i];
        const dist = calculateDistance(center, rec.point, metric);
        if (dist <= radius) {
            try results.append(allocator, .{ .id = rec.id, .distance = dist, .point = rec.point, .payload = rec.payload });
        }
    }
}

fn octreeKnnInsert(best: *std.ArrayListUnmanaged(SpatialSearchResult), item: SpatialSearchResult, k: usize, allocator: std.mem.Allocator) !void {
    var pos: usize = 0;
    while (pos < best.items.len and best.items[pos].distance <= item.distance) : (pos += 1) {}
    if (best.items.len < k) {
        try best.insert(allocator, pos, item);
    } else if (pos < k) {
        try best.insert(allocator, pos, item);
        _ = best.pop();
    }
}

fn octreeKnnWalk(
    tree: *const Octree,
    records: []const SpatialRecord3D,
    node_idx: u32,
    center: Point3D,
    k: usize,
    metric: DistanceMetric,
    best: *std.ArrayListUnmanaged(SpatialSearchResult),
    allocator: std.mem.Allocator,
) !void {
    const node = &tree.nodes.items[node_idx];
    if (best.items.len >= k) {
        const worst_distance = best.items[best.items.len - 1].distance;
        if (node.bounds.minDistanceTo(center, metric) > worst_distance) return;
    }

    if (node.children) |children| {
        for (children) |child_idx| {
            try octreeKnnWalk(tree, records, child_idx, center, k, metric, best, allocator);
        }
        return;
    }

    for (node.point_indices.items) |i| {
        const rec = records[i];
        const dist = calculateDistance(center, rec.point, metric);
        try octreeKnnInsert(best, .{ .id = rec.id, .distance = dist, .point = rec.point, .payload = rec.payload }, k, allocator);
    }
}

pub const SpatialIndex3D = struct {
    allocator: std.mem.Allocator,
    pool_alloc: ?*foundation_pool.PoolAllocator = null,
    records: std.ArrayListUnmanaged(SpatialRecord3D) = .empty,
    octree: ?Octree = null,
    octree_dirty: bool = true,

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
        if (self.octree) |*tree| tree.deinit();
    }

    fn ensureOctree(self: *SpatialIndex3D) !void {
        if (!self.octree_dirty) return;
        if (self.octree) |*tree| tree.deinit();
        self.octree = null;
        if (self.records.items.len >= OCTREE_MIN_POINTS) {
            self.octree = try Octree.build(self.allocator, self.records.items);
        }
        self.octree_dirty = false;
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
        self.octree_dirty = true;
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
        if (metric != .cosine and self.records.items.len >= OCTREE_MIN_POINTS) {
            const self_mut = @constCast(self);
            try self_mut.ensureOctree();
            if (self_mut.octree) |*tree| {
                var results: std.ArrayListUnmanaged(SpatialSearchResult) = .empty;
                errdefer results.deinit(self.allocator);
                try octreeRadiusWalk(tree, self.records.items, tree.root, center, radius, metric, &results, self.allocator);
                std.mem.sort(SpatialSearchResult, results.items, {}, sortSearchResult);
                return try results.toOwnedSlice(self.allocator);
            }
        }

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
        if (metric != .cosine and self.records.items.len >= OCTREE_MIN_POINTS) {
            const self_mut = @constCast(self);
            try self_mut.ensureOctree();
            if (self_mut.octree) |*tree| {
                var best: std.ArrayListUnmanaged(SpatialSearchResult) = .empty;
                errdefer best.deinit(self.allocator);
                try octreeKnnWalk(tree, self.records.items, tree.root, center, k, metric, &best, self.allocator);
                return try best.toOwnedSlice(self.allocator);
            }
        }

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

test "octree matches linear-scan oracle across N and distributions" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(1234);
    const random = prng.random();

    const sizes = [_]usize{ 10, 63, 64, 65, 200, 500 };
    for (sizes) |n| {
        // Uniform distribution
        {
            var index = SpatialIndex3D.init(allocator);
            defer index.deinit();
            var id: u32 = 0;
            while (id < n) : (id += 1) {
                const p = Point3D{
                    .x = random.float(f32) * 200.0 - 100.0,
                    .y = random.float(f32) * 200.0 - 100.0,
                    .z = random.float(f32) * 200.0 - 100.0,
                };
                try index.insert(id, p, "");
            }
            try assertOracleMatch(allocator, &index, random);
        }
        // Clustered distribution: three tight clusters
        {
            var index = SpatialIndex3D.init(allocator);
            defer index.deinit();
            const centers = [_]Point3D{
                .{ .x = -50, .y = -50, .z = -50 },
                .{ .x = 0, .y = 0, .z = 0 },
                .{ .x = 50, .y = 50, .z = 50 },
            };
            var id: u32 = 0;
            while (id < n) : (id += 1) {
                const c = centers[id % centers.len];
                const p = Point3D{
                    .x = c.x + random.float(f32) * 2.0 - 1.0,
                    .y = c.y + random.float(f32) * 2.0 - 1.0,
                    .z = c.z + random.float(f32) * 2.0 - 1.0,
                };
                try index.insert(id, p, "");
            }
            try assertOracleMatch(allocator, &index, random);
        }
    }
}

fn assertOracleMatch(allocator: std.mem.Allocator, index: *SpatialIndex3D, random: std.Random) !void {
    const metrics = [_]DistanceMetric{ .euclidean, .manhattan };
    for (metrics) |metric| {
        const center = Point3D{
            .x = random.float(f32) * 200.0 - 100.0,
            .y = random.float(f32) * 200.0 - 100.0,
            .z = random.float(f32) * 200.0 - 100.0,
        };

        // radiusSearch: octree path (count may or may not clear the
        // OCTREE_MIN_POINTS threshold -- either way this must match a
        // hand-rolled linear scan over the same records).
        const radius: f32 = 40.0;
        const octree_radius = try index.radiusSearch(center, radius, metric);
        defer allocator.free(octree_radius);
        const oracle_radius = try linearRadiusSearch(allocator, index.records.items, center, radius, metric);
        defer allocator.free(oracle_radius);
        try expectSameIds(oracle_radius, octree_radius);

        // nearestNeighbors
        const k: usize = 5;
        const octree_knn = try index.nearestNeighbors(center, k, metric);
        defer allocator.free(octree_knn);
        const oracle_knn = try linearNearestNeighbors(allocator, index.records.items, center, k, metric);
        defer allocator.free(oracle_knn);
        try expectSameIds(oracle_knn, octree_knn);
    }
}

fn linearRadiusSearch(allocator: std.mem.Allocator, records: []const SpatialRecord3D, center: Point3D, radius: f32, metric: DistanceMetric) ![]SpatialSearchResult {
    var results: std.ArrayListUnmanaged(SpatialSearchResult) = .empty;
    errdefer results.deinit(allocator);
    for (records) |rec| {
        const dist = calculateDistance(center, rec.point, metric);
        if (dist <= radius) {
            try results.append(allocator, .{ .id = rec.id, .distance = dist, .point = rec.point, .payload = rec.payload });
        }
    }
    std.mem.sort(SpatialSearchResult, results.items, {}, sortSearchResult);
    return try results.toOwnedSlice(allocator);
}

fn linearNearestNeighbors(allocator: std.mem.Allocator, records: []const SpatialRecord3D, center: Point3D, k: usize, metric: DistanceMetric) ![]SpatialSearchResult {
    var results: std.ArrayListUnmanaged(SpatialSearchResult) = .empty;
    errdefer results.deinit(allocator);
    for (records) |rec| {
        const dist = calculateDistance(center, rec.point, metric);
        try results.append(allocator, .{ .id = rec.id, .distance = dist, .point = rec.point, .payload = rec.payload });
    }
    std.mem.sort(SpatialSearchResult, results.items, {}, sortSearchResult);
    const n = @min(k, results.items.len);
    const owned = try allocator.alloc(SpatialSearchResult, n);
    @memcpy(owned, results.items[0..n]);
    results.deinit(allocator);
    return owned;
}

fn expectSameIds(oracle: []const SpatialSearchResult, actual: []const SpatialSearchResult) !void {
    try std.testing.expectEqual(oracle.len, actual.len);
    for (oracle, actual) |o, a| try std.testing.expectEqual(o.id, a.id);
}

test {
    std.testing.refAllDecls(@This());
}
