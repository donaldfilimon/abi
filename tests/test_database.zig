const std = @import("std");
const testing = std.testing;

// Local vector store implementation for testing
const VectorEntry = struct {
    id: []const u8,
    data: []f32,
};

const SearchResult = struct {
    id: []const u8,
    similarity: f32,
};

const VectorStore = struct {
    allocator: std.mem.Allocator,
    vectors: std.ArrayList(VectorEntry),
    dimension: usize,

    fn init(allocator: std.mem.Allocator, dim: usize) !@This() {
        const vectors = std.ArrayList(VectorEntry).initCapacity(allocator, 0) catch unreachable;
        return @This(){
            .allocator = allocator,
            .vectors = vectors,
            .dimension = dim,
        };
    }

    fn deinit(self: *@This()) void {
        for (self.vectors.items) |entry| {
            self.allocator.free(entry.data);
            self.allocator.free(entry.id);
        }
        self.vectors.deinit(self.allocator);
    }

    fn insert(self: *@This(), id: []const u8, data: []const f32) !void {
        if (data.len != self.dimension) return error.DimensionMismatch;

        const owned_id = try self.allocator.dupe(u8, id);
        const owned_data = try self.allocator.dupe(f32, data);

        try self.vectors.append(self.allocator, VectorEntry{
            .id = owned_id,
            .data = owned_data,
        });
    }

    fn search(self: *@This(), query: []const f32, k: usize) ![]SearchResult {
        if (query.len != self.dimension) return error.DimensionMismatch;

        var results = try self.allocator.alloc(SearchResult, @min(k, self.vectors.items.len));
        var result_count: usize = 0;

        for (self.vectors.items) |entry| {
            const similarity = cosineSimilarity(query, entry.data);

            if (result_count < k) {
                results[result_count] = SearchResult{
                    .id = entry.id,
                    .similarity = similarity,
                };
                result_count += 1;
            } else {
                // Find lowest similarity and replace if current is higher
                var min_idx: usize = 0;
                for (1..result_count) |i| {
                    if (results[i].similarity < results[min_idx].similarity) {
                        min_idx = i;
                    }
                }
                if (similarity > results[min_idx].similarity) {
                    results[min_idx] = SearchResult{
                        .id = entry.id,
                        .similarity = similarity,
                    };
                }
            }
        }

        // Sort results by similarity (descending)
        std.sort.insertion(SearchResult, results[0..result_count], {}, struct {
            fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                return a.similarity > b.similarity;
            }
        }.lessThan);

        return results[0..result_count];
    }

    fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;

        var dot_product: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;

        for (0..a.len) |i| {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        const magnitude = @sqrt(norm_a * norm_b);
        return if (magnitude > 0) dot_product / magnitude else 0.0;
    }
};

test "basic vector operations" {
    // Test basic vector functionality
    const vec1 = [_]f32{ 1.0, 2.0, 3.0 };
    const vec2 = [_]f32{ 4.0, 5.0, 6.0 };

    // Test dot product calculation
    var dot_product: f32 = 0.0;
    for (0..vec1.len) |i| {
        dot_product += vec1[i] * vec2[i];
    }

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try testing.expectEqual(@as(f32, 32.0), dot_product);
}

test "vector similarity calculation" {
    const allocator = testing.allocator;

    // Test vector similarity
    const Vector = struct {
        data: []f32,

        fn init(alloc: std.mem.Allocator, dims: []const f32) !@This() {
            return @This(){
                .data = try alloc.dupe(f32, dims),
            };
        }

        fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            alloc.free(self.data);
        }

        fn cosineSimilarity(self: @This(), other: @This()) f32 {
            if (self.data.len != other.data.len) return 0.0;

            var dot_product: f32 = 0.0;
            var norm_a: f32 = 0.0;
            var norm_b: f32 = 0.0;

            for (0..self.data.len) |i| {
                dot_product += self.data[i] * other.data[i];
                norm_a += self.data[i] * self.data[i];
                norm_b += other.data[i] * other.data[i];
            }

            const magnitude = @sqrt(norm_a * norm_b);
            return if (magnitude > 0) dot_product / magnitude else 0.0;
        }
    };

    // Test vector creation
    const vec1 = try Vector.init(allocator, &[_]f32{ 1.0, 0.0, 0.0 });
    defer vec1.deinit(allocator);

    const vec2 = try Vector.init(allocator, &[_]f32{ 0.0, 1.0, 0.0 });
    defer vec2.deinit(allocator);

    const vec3 = try Vector.init(allocator, &[_]f32{ 1.0, 0.0, 0.0 });
    defer vec3.deinit(allocator);

    // Test similarity calculations
    const sim_12 = vec1.cosineSimilarity(vec2); // Orthogonal vectors
    const sim_13 = vec1.cosineSimilarity(vec3); // Identical vectors

    try testing.expectApproxEqAbs(@as(f32, 0.0), sim_12, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), sim_13, 0.001);
}

test "database module: vector store creation" {
    const allocator = testing.allocator;

    // Test basic VectorStore creation
    var store = try VectorStore.init(allocator, 128); // 128 dimensions
    defer store.deinit();

    try testing.expect(store.vectors.items.len == 0);
    try testing.expect(store.dimension == 128);
}

test "database module: vector insertion and retrieval" {
    const allocator = testing.allocator;

    var store = try VectorStore.init(allocator, 4);
    defer store.deinit();

    // Test vector insertion
    const vector_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try store.insert("test_vector", &vector_data);

    try testing.expect(store.vectors.items.len == 1);
    try testing.expectEqualStrings("test_vector", store.vectors.items[0].id);
    try testing.expectEqual(@as(f32, 1.0), store.vectors.items[0].data[0]);
    try testing.expectEqual(@as(f32, 4.0), store.vectors.items[0].data[3]);
}

test "database module: vector search" {
    const allocator = testing.allocator;

    var store = try VectorStore.init(allocator, 3);
    defer store.deinit();

    // Insert test vectors
    try store.insert("vector1", &[_]f32{ 1.0, 0.0, 0.0 });
    try store.insert("vector2", &[_]f32{ 0.0, 1.0, 0.0 });
    try store.insert("vector3", &[_]f32{ 1.0, 0.0, 0.0 }); // Same as vector1

    // Search for most similar to vector1
    const query = [_]f32{ 1.0, 0.0, 0.0 };
    const results = try store.search(&query, 2);
    defer allocator.free(results);

    // Should find vector1 and vector3 as most similar
    try testing.expect(results.len == 2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), results[0].similarity, 0.001);
}

test "database module: vector indexing performance" {
    const allocator = testing.allocator;
    const num_vectors = 100;
    const dimensions = 32;

    var store = try VectorStore.init(allocator, dimensions);
    defer store.deinit();

    // Benchmark insertion
    var timer = try std.time.Timer.start();
    for (0..num_vectors) |i| {
        var vector_data: [dimensions]f32 = undefined;
        for (0..dimensions) |j| {
            vector_data[j] = @as(f32, @floatFromInt((i + j) % 10)) / 10.0;
        }

        const id = try std.fmt.allocPrint(allocator, "vector_{}", .{i});
        defer allocator.free(id);
        try store.insert(id, &vector_data);
    }
    const insert_time = timer.read();

    // Benchmark search
    const query = [_]f32{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

    timer.reset();
    const results = try store.search(&query, 10);
    defer allocator.free(results);
    const search_time = timer.read();

    std.debug.print("Database performance: {} vectors, {} dimensions\n", .{ num_vectors, dimensions });
    std.debug.print("Insert time: {}ns total, {}ns per vector\n", .{ insert_time, insert_time / num_vectors });
    std.debug.print("Search time: {}ns for top 10 results\n", .{search_time});

    try testing.expect(store.vectors.items.len == num_vectors);
    try testing.expect(results.len <= 10);
}

test "vector search performance" {
    const allocator = testing.allocator;
    const num_vectors = 50;
    const dimensions = 16;

    // Create test vectors
    var vectors = try allocator.alloc([dimensions]f32, num_vectors);
    defer allocator.free(vectors);

    // Initialize with simple pattern
    for (0..num_vectors) |i| {
        for (0..dimensions) |j| {
            vectors[i][j] = @as(f32, @floatFromInt((i + j) % 10)) / 10.0;
        }
    }

    // Simulate vector search
    const query_vector = vectors[0];
    var best_similarity: f32 = -1.0;
    var best_index: usize = 0;

    var timer = try std.time.Timer.start();
    for (1..num_vectors) |i| {
        var dot_product: f32 = 0.0;
        var norm_query: f32 = 0.0;
        var norm_candidate: f32 = 0.0;

        for (0..dimensions) |j| {
            dot_product += query_vector[j] * vectors[i][j];
            norm_query += query_vector[j] * query_vector[j];
            norm_candidate += vectors[i][j] * vectors[i][j];
        }

        const similarity = dot_product / (@sqrt(norm_query) * @sqrt(norm_candidate));
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_index = i;
        }
    }
    const search_time = timer.read();

    std.debug.print("Linear search: {} vectors, {} dimensions, {}ns\n", .{ num_vectors, dimensions, search_time });
    std.debug.print("Best match: index {}, similarity {d:.3}\n", .{ best_index, best_similarity });

    try testing.expect(best_index > 0);
    try testing.expect(best_similarity >= -1.0 and best_similarity <= 1.0);
}
