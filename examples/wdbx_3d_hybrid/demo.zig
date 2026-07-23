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
