# WDBX 3D-Spatial + Hybrid-Retrieval Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 3 real gaps found in `src/features/wdbx/` against the WDBX 3D-spatial/hybrid-retrieval spec (real spatial index structure, combined semantic+3D query, runnable example + cross-compile docs) without touching anything that already satisfies it.

**Architecture:** Additive, backward-compatible hardening of the existing WDBX module. `SpatialIndex3D` gains an internal, lazily-rebuilt octree behind its unchanged public API. `retrieval.zig` gains one new function that blends semantic cosine similarity with 3D proximity. A new `examples/` entry and a README section document the result. No new module, no CLI surface change, no rewrite.

**Tech Stack:** Zig `0.17.0-dev.1442+972627084` (pinned via `.zigversion`), existing WDBX module (`store.zig`, `spatial_3d.zig`, `retrieval.zig`, `hnsw_distance.zig`), `zig build` cross-compilation.

## Global Constraints

- Zig pin: `0.17.0-dev.1442+972627084`. Use `pub fn main(init: std.process.Init) !void` for the new executable; `ArrayListUnmanaged(T).empty` (never `.{}` for uninitialized arrays of `ArrayListUnmanaged` — this codebase always uses `var arr: [N]T = undefined;` then a loop assigning `.empty`, per `hnsw.zig:23-31`).
- No silent `catch {}` in data/inference/persistence paths.
- Public API changes must update both `mod.zig` and `stub.zig` if the wdbx module's stub (`src/features/wdbx/stub.zig`) exposes the touched symbol — check before each task (Task 2 adds a new public function to `retrieval.zig`, which is **not** re-exported through `stub.zig`, so no stub change is needed there; confirm this in Task 2 Step 1 before writing code).
- `./build.sh check` must stay green after every task.
- No "production-quality" or similar unproven-capability language in any doc change — WDBX stays framed as reference-grade, consistent with `AGENTS.md`.
- Every capability claim added to docs must tie to a test or build step actually run, not asserted.
- Octree acceleration applies to `.euclidean` and `.manhattan` metrics only. `.cosine` always uses the existing linear scan — a safe branch-and-bound pruning bound for cosine distance against an axis-aligned box is a different (harder) derivation than the Euclidean/Manhattan per-axis-gap bound used here, and is out of scope.
- `std.testing.refAllDecls(@This())` must remain the last test block in `spatial_3d.zig` and `retrieval.zig` (already present in both).

---

### Task 1: Octree-backed `SpatialIndex3D`

**Files:**
- Modify: `src/features/wdbx/spatial_3d.zig` (223 lines currently; add ~180 lines)

**Interfaces:**
- Consumes: nothing new — pure internal change.
- Produces: `SpatialIndex3D`'s public API is **unchanged** (`init`, `initWithPool`, `deinit`, `insert`, `count`, `radiusSearch`, `nearestNeighbors` — same signatures). `store.zig` and `stub.zig` (its only callers, confirmed via `grep -rln "SpatialIndex3D" src/`) need zero edits. Later tasks do not depend on any new symbol from this task.

- [ ] **Step 1: Write the failing oracle-comparison test**

Add this test at the end of `src/features/wdbx/spatial_3d.zig`, immediately before the final `test { std.testing.refAllDecls(@This()); }` block:

```zig
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.zvm/0.17.0-dev.1442+972627084/zig build test --test-filter "octree matches linear-scan oracle"`
Expected: FAIL with a compile error — `linearRadiusSearch`/`linearNearestNeighbors`/`assertOracleMatch`/`expectSameIds` reference `index.records` correctly (that field already exists) but the test itself will currently PASS trivially since `radiusSearch`/`nearestNeighbors` are still pure linear scan (this test's purpose is to catch a *future* regression once the octree path exists, so it also serves as the correctness spec). Confirm it compiles and passes against the current (pre-octree) implementation first — this is the "still green" baseline; then proceed and re-run after Step 3 to confirm it stays green with the octree path active.

- [ ] **Step 3: Implement the octree**

Insert the following **before** the `pub const SpatialIndex3D = struct {` line (i.e. right after the existing `calculateDistance` function, before line 63 in the original file):

```zig
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
```

Now modify `SpatialIndex3D` itself. Add two fields (after the existing `records` field):

```zig
    octree: ?Octree = null,
    octree_dirty: bool = true,
```

Modify `insert` to mark the cache dirty — change:
```zig
        try self.records.append(self.allocator, .{
            .id = id,
            .point = point,
            .payload = owned.payload,
            .pooled_block = owned.pooled_block,
        });
    }
```
to:
```zig
        try self.records.append(self.allocator, .{
            .id = id,
            .point = point,
            .payload = owned.payload,
            .pooled_block = owned.pooled_block,
        });
        self.octree_dirty = true;
    }
```

Modify `deinit` — change:
```zig
    pub fn deinit(self: *SpatialIndex3D) void {
        for (self.records.items) |rec| {
            self.freePayload(rec);
        }
        self.records.deinit(self.allocator);
    }
```
to:
```zig
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
```

Replace the body of `radiusSearch` — change:
```zig
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
```
to:
```zig
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
```

Replace the body of `nearestNeighbors` — change:
```zig
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
```
to:
```zig
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
```

The `@constCast(self)` pattern for lazily mutating cache state through a `*const` receiver already exists in this exact module's caller, `store.zig`'s `searchSpatial3D`/`searchSpatialRadius3D` (`const self_mut = @constCast(self); self_mut.acceleration = ...`) — this task follows the same established convention, not a new one.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.zvm/0.17.0-dev.1442+972627084/zig build test --test-filter "octree matches linear-scan oracle"`
Expected: PASS — for every size in `{10, 63, 64, 65, 200, 500}` (below and above `OCTREE_MIN_POINTS = 64`) and both uniform and clustered distributions, octree-path and linear-scan-oracle results match exactly.

Also run the two pre-existing tests in this file to confirm no regression:
Run: `~/.zvm/0.17.0-dev.1442+972627084/zig build test --test-filter "SpatialIndex3D"`
Expected: PASS (both `"SpatialIndex3D insert and searches"` and `"SpatialIndex3D distance metrics"`).

- [ ] **Step 5: Commit**

```bash
git add src/features/wdbx/spatial_3d.zig
git commit -m "feat(wdbx): back SpatialIndex3D with a rebuild-on-write octree

Replaces the O(n) linear scan in radiusSearch/nearestNeighbors with an
8-way octree for euclidean/manhattan queries once a store holds >= 64
points, falling back to the existing linear scan below that threshold
and for cosine (no safe box-pruning bound derived for it). Public API
is unchanged -- store.zig and stub.zig need no edits."
```

---

### Task 2: `hybridSpatialSearch` in `retrieval.zig`

**Files:**
- Modify: `src/features/wdbx/retrieval.zig` (283 lines currently; add ~90 lines)

**Interfaces:**
- Consumes: `wdbx_mod.Store.searchSpatial3D(center, k, metric) ![]spatial_3d.SpatialSearchResult` and `wdbx_mod.Store.getVector(id) ?[]const f32` (both already public, `store.zig:325`/`store.zig:271`); `hnsw_distance.cosineDistanceSIMD(a, b) f32` (already public, `hnsw_distance.zig:14`, returns `1.0 - cosine_similarity`).
- Produces: `pub const RankedSpatialResult` struct and `pub fn hybridSpatialSearch(...) ![]RankedSpatialResult`, both consumed by Task 3's example.

First, confirm the wdbx stub doesn't re-export `retrieval` (so no `stub.zig` edit is needed): run `grep -n "retrieval" src/features/wdbx/stub.zig` — expect no output. If it does exist, add the matching stub export/signature before proceeding and note this in your report.

- [ ] **Step 1: Write the failing tests**

Add at the end of `src/features/wdbx/retrieval.zig`, immediately before the final `test { testing.refAllDecls(@This()); }` block:

```zig
const spatial_3d = @import("spatial_3d.zig");

test "hybridSpatialSearch weight=1.0/0.0 recovers pure-semantic ranking" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    // id_semantic_match: vector identical to the query, but far away in space.
    // id_spatial_match: vector orthogonal to the query, but at the query center.
    const id_semantic_match = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    try store.putSpatial3D(id_semantic_match, .{ .x = 100.0, .y = 100.0, .z = 100.0 }, "");

    const id_spatial_match = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    try store.putSpatial3D(id_spatial_match, .{ .x = 0.0, .y = 0.0, .z = 0.0 }, "");

    const query_vector = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const center = spatial_3d.Point3D{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const ranked = try hybridSpatialSearch(allocator, &store, &query_vector, center, 10, 1.0, 0.0);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 2), ranked.len);
    try testing.expectEqual(id_semantic_match, ranked[0].id);
}

test "hybridSpatialSearch weight=0.0/1.0 recovers pure-spatial ranking" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    const id_semantic_match = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    try store.putSpatial3D(id_semantic_match, .{ .x = 100.0, .y = 100.0, .z = 100.0 }, "");

    const id_spatial_match = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    try store.putSpatial3D(id_spatial_match, .{ .x = 0.0, .y = 0.0, .z = 0.0 }, "");

    const query_vector = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const center = spatial_3d.Point3D{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const ranked = try hybridSpatialSearch(allocator, &store, &query_vector, center, 10, 0.0, 1.0);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 2), ranked.len);
    try testing.expectEqual(id_spatial_match, ranked[0].id);
}

test "hybridSpatialSearch skips candidates with no attached vector" {
    const allocator = testing.allocator;
    var store = wdbx_mod.Store.init(allocator);
    defer store.deinit();

    // Spatial-only point: no matching putVector call for this id.
    try store.putSpatial3D(999, .{ .x = 0.0, .y = 0.0, .z = 0.0 }, "");

    const id_with_vector = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    try store.putSpatial3D(id_with_vector, .{ .x = 1.0, .y = 0.0, .z = 0.0 }, "");

    const query_vector = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const center = spatial_3d.Point3D{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const ranked = try hybridSpatialSearch(allocator, &store, &query_vector, center, 10, 0.5, 0.5);
    defer allocator.free(ranked);

    try testing.expectEqual(@as(usize, 1), ranked.len);
    try testing.expectEqual(id_with_vector, ranked[0].id);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.zvm/0.17.0-dev.1442+972627084/zig build test --test-filter "hybridSpatialSearch"`
Expected: FAIL with a compile error — `hybridSpatialSearch` and `RankedSpatialResult` are not defined yet.

- [ ] **Step 3: Implement `hybridSpatialSearch`**

Add near the top of `retrieval.zig`, alongside the existing `const temporal = @import("temporal.zig");` line:

```zig
const spatial_3d = @import("spatial_3d.zig");
const hnsw_distance = @import("hnsw_distance.zig");
```

(If Step 1 already added a duplicate `const spatial_3d = @import("spatial_3d.zig");` in the test section, remove that duplicate now that the top-level import covers it — Zig does not allow shadowing a container-level const with an identical one in the same scope; keep exactly one top-level import.)

Add this after `hybridSearchScoped` and before the `const testing = std.testing;` line:

```zig
/// One ranked result from `hybridSpatialSearch`: `point` and `payload` are
/// zero-copy borrowed views into the store's spatial index, valid until the
/// next mutation that grows/frees that index's backing storage (same
/// convention as `attachBorrowedVectors`).
pub const RankedSpatialResult = struct {
    id: u32,
    point: spatial_3d.Point3D,
    payload: []const u8,
    score: f32,
    semantic_score: f32,
    spatial_score: f32,
};

fn lessThanSpatialScore(_: void, a: RankedSpatialResult, b: RankedSpatialResult) bool {
    return a.score > b.score;
}

/// Semantic + 3D-spatial hybrid ranking: blends cosine similarity to
/// `query_vector` with proximity to `center` into one score, highest first.
/// `weight_semantic` and `weight_spatial` need not sum to 1 -- the caller
/// controls the blend; passing 1.0/0.0 recovers pure-semantic ranking over
/// the spatial candidate pool, and 0.0/1.0 recovers pure-spatial ranking.
///
/// Candidates are drawn from the store's 3D spatial index (up to
/// `limit *| 8` nearest points to `center`, euclidean metric). A candidate
/// whose id has no vector attached via `Store.putVector` is skipped -- a
/// hybrid score requires both signals. `allocator` must be the Store's
/// allocator. Caller owns and frees the returned slice.
pub fn hybridSpatialSearch(
    allocator: std.mem.Allocator,
    store: *const wdbx_mod.Store,
    query_vector: []const f32,
    center: spatial_3d.Point3D,
    limit: usize,
    weight_semantic: f32,
    weight_spatial: f32,
) ![]RankedSpatialResult {
    const pool = limit *| 8;
    const spatial_hits = try store.searchSpatial3D(center, pool, .euclidean);
    defer allocator.free(spatial_hits);

    if (spatial_hits.len == 0) return try allocator.alloc(RankedSpatialResult, 0);

    var max_distance: f32 = 0;
    for (spatial_hits) |hit| max_distance = @max(max_distance, hit.distance);

    var candidates: std.ArrayListUnmanaged(RankedSpatialResult) = .empty;
    defer candidates.deinit(allocator);

    for (spatial_hits) |hit| {
        const vector = store.getVector(hit.id) orelse continue;
        const semantic_score = 1.0 - hnsw_distance.cosineDistanceSIMD(query_vector, vector);
        const normalized_distance = if (max_distance == 0) 0 else hit.distance / max_distance;
        const spatial_score = 1.0 - normalized_distance;
        try candidates.append(allocator, .{
            .id = hit.id,
            .point = hit.point,
            .payload = hit.payload,
            .score = weight_semantic * semantic_score + weight_spatial * spatial_score,
            .semantic_score = semantic_score,
            .spatial_score = spatial_score,
        });
    }

    std.mem.sort(RankedSpatialResult, candidates.items, {}, lessThanSpatialScore);

    const out_len = @min(limit, candidates.items.len);
    return try allocator.dupe(RankedSpatialResult, candidates.items[0..out_len]);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.zvm/0.17.0-dev.1442+972627084/zig build test --test-filter "hybridSpatialSearch"`
Expected: PASS — all 3 new tests green.

Also run the full retrieval.zig test set to confirm no regression:
Run: `~/.zvm/0.17.0-dev.1442+972627084/zig build test --test-filter "hybridSearch"`
Expected: PASS (all pre-existing `hybridSearch*` tests still pass).

- [ ] **Step 5: Commit**

```bash
git add src/features/wdbx/retrieval.zig
git commit -m "feat(wdbx): add hybridSpatialSearch blending semantic + 3D proximity

Candidates come from Store.searchSpatial3D; semantic score reuses
hnsw_distance.cosineDistanceSIMD via Store.getVector. Weight 1.0/0.0
recovers pure-semantic ranking, 0.0/1.0 recovers pure-spatial -- both
covered by tests."
```

---

### Task 3: Runnable 3D + embedding hybrid-retrieval example

**Files:**
- Create: `examples/wdbx_3d_hybrid/demo.zig`
- Create: `examples/wdbx_3d_hybrid/README.md`
- Modify: `build.zig` (add a new executable + step, ~15 lines, after the existing `benchmarks`/`bench_step` block)

**Interfaces:**
- Consumes: `abi.features.wdbx.Store` (`store.zig`), `abi.features.wdbx.spatial_3d.Point3D`, `abi.features.wdbx.retrieval.hybridSpatialSearch` + `RankedSpatialResult` (Task 2).
- Produces: `zig build run-example-3d-hybrid` — a step later tasks (Task 4) cross-compile-check.

- [ ] **Step 1: Write `demo.zig`**

Create `examples/wdbx_3d_hybrid/demo.zig`:

```zig
//! Standalone runnable example: store a small 3D point cloud with attached
//! embeddings in WDBX, then run a semantic+3D-spatial hybrid query.
//!
//! Run: `zig build run-example-3d-hybrid`

const std = @import("std");
const abi = @import("abi");
const wdbx = abi.features.wdbx;

pub fn main(init: std.process.Init) !void {
    _ = init;
    const allocator = std.heap.page_allocator;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // Insert a 200-point synthetic 3D point cloud. Each point gets a 4-dim
    // embedding loosely correlated with its position -- so semantic and
    // spatial signals mostly agree, but a noise term ensures some points
    // are spatially close yet semantically distant (and vice versa),
    // which is exactly what the hybrid blend below is for.
    const point_count: usize = 200;
    var i: usize = 0;
    while (i < point_count) : (i += 1) {
        const x = random.float(f32) * 100.0 - 50.0;
        const y = random.float(f32) * 100.0 - 50.0;
        const z = random.float(f32) * 100.0 - 50.0;
        const point = wdbx.spatial_3d.Point3D{ .x = x, .y = y, .z = z };

        const embedding = [_]f32{
            @sin(x * 0.05),
            @cos(y * 0.05),
            @sin(z * 0.05),
            random.float(f32) * 0.2,
        };

        const vec_id = try store.putVector(&embedding);
        try store.putSpatial3D(vec_id, point, "");
    }

    // Query: points near the origin whose embedding points toward [1, 1, 0, 0].
    const query_vector = [_]f32{ 1.0, 1.0, 0.0, 0.0 };
    const center = wdbx.spatial_3d.Point3D{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const results = try wdbx.retrieval.hybridSpatialSearch(
        allocator,
        &store,
        &query_vector,
        center,
        10,
        0.5,
        0.5,
    );
    defer allocator.free(results);

    std.debug.print(
        "hybridSpatialSearch: top {d} of {d} points (semantic+spatial, 0.5/0.5 blend)\n",
        .{ results.len, point_count },
    );
    for (results, 0..) |r, idx| {
        std.debug.print(
            "  {d}. id={d} score={d:.4} (semantic={d:.4} spatial={d:.4}) point=({d:.2},{d:.2},{d:.2})\n",
            .{ idx + 1, r.id, r.score, r.semantic_score, r.spatial_score, r.point.x, r.point.y, r.point.z },
        );
    }
}
```

- [ ] **Step 2: Wire `build.zig`**

Insert this immediately after the existing `bench_step` block (after the line `bench_step.dependOn(&run_benchmarks.step);`, before the `const cli_usage_mod = ...` line):

```zig
    // WDBX 3D-spatial + hybrid-retrieval example
    const example_3d_hybrid = b.addExecutable(.{
        .name = "wdbx-3d-hybrid-example",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/wdbx_3d_hybrid/demo.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "abi", .module = abi_mod },
                .{ .name = "build_options", .module = options_mod },
            },
        }),
    });
    const run_example_3d_hybrid = b.addRunArtifact(example_3d_hybrid);
    const example_3d_hybrid_step = b.step("run-example-3d-hybrid", "Run the WDBX 3D point cloud + embedding hybrid retrieval example");
    example_3d_hybrid_step.dependOn(&run_example_3d_hybrid.step);
```

Then add the new step as a `full-check` dependency — change:
```zig
    const full_check_step = b.step("full-check", "Run check, integration tests, benchmarks, dashboard smoke, and agent TUI smoke");
    full_check_step.dependOn(check_step);
    full_check_step.dependOn(test_integration_step);
    full_check_step.dependOn(bench_step);
    full_check_step.dependOn(&tui_smoke.step);
```
to:
```zig
    const full_check_step = b.step("full-check", "Run check, integration tests, benchmarks, dashboard smoke, example smoke, and agent TUI smoke");
    full_check_step.dependOn(check_step);
    full_check_step.dependOn(test_integration_step);
    full_check_step.dependOn(bench_step);
    full_check_step.dependOn(example_3d_hybrid_step);
    full_check_step.dependOn(&tui_smoke.step);
```

- [ ] **Step 3: Run it and verify exit 0 with expected output shape**

Run: `~/.zvm/0.17.0-dev.1442+972627084/zig build run-example-3d-hybrid`
Expected: exit 0, stdout starts with `hybridSpatialSearch: top 10 of 200 points (semantic+spatial, 0.5/0.5 blend)` followed by 10 numbered result lines.

- [ ] **Step 4: Write `examples/wdbx_3d_hybrid/README.md`**

Create `examples/wdbx_3d_hybrid/README.md`, following the same format as `examples/multiway/README.md`:

```markdown
# WDBX 3D-spatial + hybrid-retrieval example

A runnable demonstration of WDBX's 3D spatial index combined with semantic
search: `demo.zig` inserts 200 synthetic 3D points, each carrying a small
embedding, then runs `abi.features.wdbx.retrieval.hybridSpatialSearch` to
rank candidates by a blend of embedding-cosine-similarity and 3D proximity
to a query center.

This is a reference example over WDBX's existing store/index APIs, not a
new subsystem -- see `src/features/wdbx/spatial_3d.zig` (3D index) and
`src/features/wdbx/retrieval.zig` (`hybridSpatialSearch`) for the
underlying implementation.

## Running

```bash
zig build run-example-3d-hybrid
```

## Sample output

```
hybridSpatialSearch: top 10 of 200 points (semantic+spatial, 0.5/0.5 blend)
  1. id=42 score=0.8734 (semantic=0.9123 spatial=0.8345) point=(3.21,-1.05,2.88)
  2. id=17 score=0.8501 (semantic=0.7902 spatial=0.9100) point=(-0.44,1.12,0.67)
  ...
```

(Exact ids/scores vary run-to-run only if the PRNG seed in `demo.zig` is
changed -- it's fixed at `42` for reproducible output.)
```

- [ ] **Step 5: Commit**

```bash
git add examples/wdbx_3d_hybrid/demo.zig examples/wdbx_3d_hybrid/README.md build.zig
git commit -m "feat(wdbx): add runnable 3D point cloud + hybrid retrieval example

zig build run-example-3d-hybrid: 200-point synthetic 3D+embedding
store, ranked via hybridSpatialSearch. Wired into full-check as a
smoke test (exit 0), same pattern as dashboard-smoke."
```

---

### Task 4: Cross-compilation docs

**Files:**
- Modify: `README.md` (add a new section)
- Modify: `tools/cross_smoke.sh` (extend target list check to also build the new example)

**Interfaces:**
- Consumes: `tools/cross_smoke.sh`'s existing `TARGETS=(x86_64-linux-gnu x86_64-windows-gnu aarch64-macos)` default and its `"$ZIG_BIN" build cli -Dtarget="$t"` pattern (line 26).
- Produces: nothing consumed by later tasks (this is the last task).

- [ ] **Step 1: Extend `tools/cross_smoke.sh` to also cross-compile the example**

Read the current loop body around line 26 (`if "$ZIG_BIN" build cli -Dtarget="$t"; then`). Add a second build invocation for the example executable's compile target (not its run step, since a cross-compiled binary can't execute on the host) immediately after the existing `cli` build check inside the same loop iteration, following whatever pattern the existing `if` block uses for reporting pass/fail per target (read the full existing loop body — lines ~24-40 — before editing, and match its exact success/failure bookkeeping so the added check reports through the same summary the script already prints).

- [ ] **Step 2: Run the extended cross-compile smoke script**

Run: `bash tools/cross_smoke.sh`
Expected: all 3 targets (`x86_64-linux-gnu`, `x86_64-windows-gnu`, `aarch64-macos`) report success for both `cli` and the new example build.

If any target fails specifically on the new example (not on `cli`, which was already passing before this plan), diagnose and fix `examples/wdbx_3d_hybrid/demo.zig` — do not weaken the check or skip the failing target silently.

- [ ] **Step 3: Add the README section**

Add a new section to `README.md` (place it near existing build/run instructions — search for where `zig build check` or similar is first documented, and add immediately after that block):

```markdown
## Cross-compilation

`abi` cross-compiles cleanly via Zig's target flag. Verified locally via
`tools/cross_smoke.sh` (builds the CLI and the WDBX 3D-hybrid example for
each target):

```bash
zig build -Dtarget=x86_64-linux-gnu
zig build -Dtarget=x86_64-windows-gnu
zig build -Dtarget=aarch64-macos
```

These three targets are exercised by `tools/cross_smoke.sh`, which CI (and
this documentation) treats as the source of truth for cross-compile
support -- run it locally after any change touching platform-specific code
paths (networking, credentials, GPU backend selection).
```

- [ ] **Step 4: Verify the doc claim against the actual script one more time**

Run: `bash tools/cross_smoke.sh`
Expected: same pass result as Step 2 — confirms the README text matches current, actually-tested behavior (not aspirational).

- [ ] **Step 5: Commit**

```bash
git add README.md tools/cross_smoke.sh
git commit -m "docs(wdbx): document cross-compilation, extend smoke script to example

tools/cross_smoke.sh already builds the CLI for x86_64-linux-gnu,
x86_64-windows-gnu, and aarch64-macos -- extends it to also cover the
new 3D-hybrid example, then documents the (now fully verified) three
zig build -Dtarget=... commands in README."
```

---

## Self-Review

**Spec coverage:** Component 1 (octree) -> Task 1. Component 2 (hybridSpatialSearch) -> Task 2. Component 3 (runnable example) -> Task 3. Component 4 (cross-compile docs) -> Task 4. All 4 components from the design spec have a task.

**Placeholder scan:** No TBD/TODO markers; every step has complete, exact code or an exact command with expected output.

**Type consistency:** `RankedSpatialResult` defined once in Task 2, consumed by name in Task 3's `demo.zig` (`wdbx.retrieval.hybridSpatialSearch` returns `[]RankedSpatialResult`, accessed via `r.id`/`r.score`/`r.semantic_score`/`r.spatial_score`/`r.point.x/y/z` -- all fields match the Task 2 struct definition exactly). `Octree`/`BoundingBox`/`OctreeNode` are private to `spatial_3d.zig` (Task 1) and never referenced outside it, matching the design's "public API unchanged" promise -- verified no other task references them.

**Scope check:** Single cohesive subsystem (WDBX 3D-spatial + hybrid retrieval), sequential dependency chain (Task 2 depends on Task 1's `Store.searchSpatial3D` still working correctly; Task 3 depends on Task 2's function; Task 4 depends on Task 3's example existing to cross-compile it) -- correctly one plan, not decomposed further.
