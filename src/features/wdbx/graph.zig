const std = @import("std");

/// A node in the HNSW graph.
pub const Node = struct {
    id: u64,
    vector: []f32,
    layers: std.ArrayList(std.ArrayList(u64)), // Adjacency lists for each layer

    pub fn init(allocator: std.mem.Allocator, id: u64, vector: []const f32, max_layers: usize) !Node {
        var layers = std.ArrayList(std.ArrayList(u64)).init(allocator);
        for (0..max_layers) |_| {
            try layers.append(std.ArrayList(u64).init(allocator));
        }
        const vec = try allocator.dupe(f32, vector);
        return .{
            .id = id,
            .vector = vec,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Node, allocator: std.mem.Allocator) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit();
        allocator.free(self.vector);
    }
};

/// Simple Euclidean distance metric.
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |va, vb| {
        sum += std.math.pow(f32, va - vb, 2);
    }
    return @sqrt(sum);
}
