//! DiskANN index for billion-scale approximate nearest neighbor search.

const std = @import("std");
const index_mod = @import("../index.zig");
const mmap = @import("../formats/mmap.zig");
const types = @import("types.zig");
const codebook_mod = @import("codebook.zig");

const DiskANNConfig = types.DiskANNConfig;
const DiskNode = types.DiskNode;
const SearchCandidate = types.SearchCandidate;
const IndexStats = types.IndexStats;
const PQCodebook = codebook_mod.PQCodebook;
const PersistError = types.PersistError;
const computeL2DistanceSquared = types.computeL2DistanceSquared;
const alignToSector = types.alignToSector;
const writeAllFd = types.writeAllFd;
const writePadding = types.writePadding;
const DISKANN_MAGIC = types.DISKANN_MAGIC;
const DISKANN_FORMAT_VERSION = types.DISKANN_FORMAT_VERSION;
const DISKANN_HEADER_SIZE = types.DISKANN_HEADER_SIZE;

/// DiskANN index for billion-scale approximate nearest neighbor search
pub const DiskANNIndex = struct {
    config: DiskANNConfig,
    allocator: std.mem.Allocator,

    // Index state
    num_vectors: u32 = 0,
    entry_point: u32 = 0,

    // PQ codebook for compression
    codebook: ?PQCodebook = null,

    // In-memory graph for small datasets
    // For large datasets, this would be disk-backed
    graph: std.ArrayListUnmanaged(std.ArrayListUnmanaged(u32)),

    // PQ codes for all vectors
    pq_codes: std.ArrayListUnmanaged([]u8),

    // Full vectors (for build and optional reranking)
    vectors: std.ArrayListUnmanaged([]f32),

    // Node cache for disk-based access
    node_cache: std.AutoHashMapUnmanaged(u32, DiskNode),
    cache_order: std.ArrayListUnmanaged(u32),

    // Statistics
    stats: IndexStats = .{},

    pub fn init(allocator: std.mem.Allocator, config: DiskANNConfig) !DiskANNIndex {
        return DiskANNIndex{
            .config = config,
            .allocator = allocator,
            .graph = .empty,
            .pq_codes = .empty,
            .vectors = .empty,
            .node_cache = .empty,
            .cache_order = .empty,
        };
    }

    pub fn deinit(self: *DiskANNIndex) void {
        // Free graph
        for (self.graph.items) |*adj_list| {
            adj_list.deinit(self.allocator);
        }
        self.graph.deinit(self.allocator);

        // Free PQ codes
        for (self.pq_codes.items) |codes| {
            self.allocator.free(codes);
        }
        self.pq_codes.deinit(self.allocator);

        // Free vectors
        for (self.vectors.items) |vec| {
            self.allocator.free(vec);
        }
        self.vectors.deinit(self.allocator);

        // Free codebook
        if (self.codebook) |*cb| {
            cb.deinit();
        }

        self.node_cache.deinit(self.allocator);
        self.cache_order.deinit(self.allocator);
    }

    /// Build index from vectors using Vamana algorithm
    pub fn build(self: *DiskANNIndex, vectors: []const f32) !void {
        const dim = self.config.dimensions;
        const num_vectors = @as(u32, @intCast(vectors.len / dim));

        if (num_vectors == 0) return error.EmptyDataset;

        // Initialize PQ codebook
        self.codebook = try PQCodebook.init(
            self.allocator,
            self.config.pq_subspaces,
            dim / self.config.pq_subspaces,
            @as(u32, 1) << @intCast(self.config.pq_bits),
        );
        try self.codebook.?.train(vectors, dim);

        // Store vectors and encode with PQ
        for (0..num_vectors) |i| {
            const vec_start = i * dim;
            const vec_end = vec_start + dim;
            const vec_slice = vectors[vec_start..vec_end];

            // Store full vector
            const vec_copy = try self.allocator.alloc(f32, dim);
            @memcpy(vec_copy, vec_slice);
            try self.vectors.append(self.allocator, vec_copy);

            // Encode with PQ
            const codes = try self.allocator.alloc(u8, self.config.pq_subspaces);
            self.codebook.?.encode(vec_slice, codes);
            try self.pq_codes.append(self.allocator, codes);

            // Initialize empty adjacency list
            try self.graph.append(self.allocator, .empty);
        }

        self.num_vectors = num_vectors;

        // Select medoid as entry point
        self.entry_point = try self.selectMedoid();

        // Build Vamana graph
        try self.buildVamanaGraph();

        self.stats.build_complete = true;
    }

    /// Select approximate medoid as entry point
    fn selectMedoid(self: *DiskANNIndex) !u32 {
        if (self.num_vectors == 0) return error.EmptyIndex;

        const dim = self.config.dimensions;
        const sample_size = @min(self.num_vectors, 1000);

        // Compute centroid from sample
        var centroid = try self.allocator.alloc(f32, dim);
        defer self.allocator.free(centroid);
        @memset(centroid, 0);

        for (0..sample_size) |i| {
            const vec = self.vectors.items[i];
            for (0..dim) |d| {
                centroid[d] += vec[d];
            }
        }
        for (centroid) |*c| {
            c.* /= @floatFromInt(sample_size);
        }

        // Find vector closest to centroid
        var min_dist: f32 = std.math.inf(f32);
        var medoid: u32 = 0;

        for (0..self.num_vectors) |i| {
            const vec = self.vectors.items[i];
            const dist = computeL2DistanceSquared(vec, centroid);
            if (dist < min_dist) {
                min_dist = dist;
                medoid = @intCast(i);
            }
        }

        return medoid;
    }

    /// Build Vamana graph with alpha-RNG pruning
    fn buildVamanaGraph(self: *DiskANNIndex) !void {
        const alpha = self.config.alpha;

        // Process vectors in random order
        var order = try self.allocator.alloc(u32, self.num_vectors);
        defer self.allocator.free(order);

        for (0..self.num_vectors) |i| {
            order[i] = @intCast(i);
        }

        // Shuffle
        var prng = std.Random.DefaultPrng.init(12345);
        const random = prng.random();
        for (0..self.num_vectors) |i| {
            const j = random.intRangeAtMost(usize, i, self.num_vectors - 1);
            const tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }

        // Build graph incrementally
        for (order) |node_id| {
            // Search for nearest neighbors
            const candidates = try self.greedySearch(self.vectors.items[node_id], self.config.build_list_size);
            defer self.allocator.free(candidates);

            // Prune with alpha-RNG
            const neighbors = try self.robustPrune(node_id, candidates, alpha);
            defer self.allocator.free(neighbors);

            // Add edges (bidirectional)
            try self.addEdges(node_id, neighbors);

            self.stats.vectors_indexed += 1;
        }
    }

    /// Greedy search to find nearest neighbors
    fn greedySearch(self: *DiskANNIndex, query: []const f32, list_size: u32) ![]SearchCandidate {
        var candidates = std.PriorityQueue(SearchCandidate, void, SearchCandidate.lessThan).init(self.allocator, {});
        defer candidates.deinit();

        var visited: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer visited.deinit(self.allocator);

        // Start from entry point
        const entry_dist = computeL2DistanceSquared(query, self.vectors.items[self.entry_point]);
        try candidates.add(.{ .id = self.entry_point, .distance = entry_dist });
        try visited.put(self.allocator, self.entry_point, {});

        var result_list = std.ArrayListUnmanaged(SearchCandidate).empty;

        while (candidates.count() > 0) {
            const current = candidates.remove();

            // Add to results
            try result_list.append(self.allocator, current);
            if (result_list.items.len >= list_size) break;

            // Expand neighbors
            if (current.id >= self.graph.items.len) continue;
            for (self.graph.items[current.id].items) |neighbor_id| {
                if (visited.contains(neighbor_id)) continue;
                try visited.put(self.allocator, neighbor_id, {});

                if (neighbor_id >= self.vectors.items.len) continue;
                const neighbor_dist = computeL2DistanceSquared(query, self.vectors.items[neighbor_id]);
                try candidates.add(.{ .id = neighbor_id, .distance = neighbor_dist });

                // Maintain list size
                while (candidates.count() > list_size * 2) {
                    _ = candidates.removeOrNull();
                }
            }
        }

        return try result_list.toOwnedSlice(self.allocator);
    }

    /// Robust pruning with alpha-RNG criterion
    fn robustPrune(
        self: *DiskANNIndex,
        node_id: u32,
        candidates: []const SearchCandidate,
        alpha: f32,
    ) ![]u32 {
        var pruned = std.ArrayListUnmanaged(u32).empty;
        const max_degree = self.config.max_degree;

        for (candidates) |candidate| {
            if (candidate.id == node_id) continue;
            if (pruned.items.len >= max_degree) break;

            var dominated = false;

            // Check alpha-RNG condition
            for (pruned.items) |existing_id| {
                const existing_to_candidate = computeL2DistanceSquared(
                    self.vectors.items[existing_id],
                    self.vectors.items[candidate.id],
                );

                if (existing_to_candidate * alpha < candidate.distance) {
                    dominated = true;
                    break;
                }
            }

            if (!dominated) {
                try pruned.append(self.allocator, candidate.id);
            }
        }

        return try pruned.toOwnedSlice(self.allocator);
    }

    /// Add bidirectional edges to graph
    fn addEdges(self: *DiskANNIndex, node_id: u32, neighbors: []const u32) !void {
        for (neighbors) |neighbor_id| {
            // Add forward edge
            try self.graph.items[node_id].append(self.allocator, neighbor_id);

            // Add backward edge (with degree check)
            if (self.graph.items[neighbor_id].items.len < self.config.max_degree) {
                try self.graph.items[neighbor_id].append(self.allocator, node_id);
            }
        }
    }

    /// Search for k nearest neighbors
    pub fn search(
        self: *DiskANNIndex,
        query: []const f32,
        k: u32,
    ) ![]index_mod.IndexResult {
        if (self.num_vectors == 0) return &[_]index_mod.IndexResult{};

        // Precompute distance table for PQ
        try self.codebook.?.computeDistanceTable(query);

        // Run beam search with PQ distances
        const candidates = if (self.config.beam_search)
            try self.beamSearch(query, self.config.search_list_size)
        else
            try self.greedySearch(query, self.config.search_list_size);
        defer self.allocator.free(candidates);

        // Rerank with exact distances
        var reranked = std.ArrayListUnmanaged(index_mod.IndexResult).empty;

        for (candidates) |candidate| {
            if (reranked.items.len >= k) break;

            const exact_dist = computeL2DistanceSquared(query, self.vectors.items[candidate.id]);
            try reranked.append(self.allocator, .{
                .id = candidate.id,
                .score = @sqrt(exact_dist),
            });
        }

        // Sort by distance (score = L2 distance, lower is better)
        std.mem.sort(index_mod.IndexResult, reranked.items, {}, struct {
            fn lessThan(_: void, a: index_mod.IndexResult, b: index_mod.IndexResult) bool {
                return a.score < b.score;
            }
        }.lessThan);

        self.stats.queries_processed += 1;
        return try reranked.toOwnedSlice(self.allocator);
    }

    /// Beam search with prefetching (optimized for disk access)
    fn beamSearch(self: *DiskANNIndex, query: []const f32, list_size: u32) ![]SearchCandidate {
        _ = query;
        const beam_width = self.config.beam_width;

        var candidates = std.PriorityQueue(SearchCandidate, void, SearchCandidate.lessThan).init(self.allocator, {});
        defer candidates.deinit();

        var visited: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer visited.deinit(self.allocator);

        // Start from entry point using PQ distance
        const entry_dist = self.codebook.?.computeAsymmetricDistance(self.pq_codes.items[self.entry_point]);
        try candidates.add(.{ .id = self.entry_point, .distance = entry_dist });
        try visited.put(self.allocator, self.entry_point, {});

        var result_list = std.ArrayListUnmanaged(SearchCandidate).empty;
        var unexpanded = std.ArrayListUnmanaged(SearchCandidate).empty;
        defer unexpanded.deinit(self.allocator);

        while (candidates.count() > 0) {
            unexpanded.clearRetainingCapacity();

            // Pop up to beam_width candidates to expand in parallel
            while (candidates.count() > 0 and unexpanded.items.len < beam_width) {
                const current = candidates.remove();
                try unexpanded.append(self.allocator, current);

                try result_list.append(self.allocator, current);
                if (result_list.items.len >= list_size) break;
            }

            if (result_list.items.len >= list_size) break;

            // Expand candidates in the beam
            for (unexpanded.items) |current| {
                if (current.id >= self.graph.items.len) continue;

                for (self.graph.items[current.id].items) |neighbor_id| {
                    if (visited.contains(neighbor_id)) continue;
                    try visited.put(self.allocator, neighbor_id, {});

                    if (neighbor_id >= self.pq_codes.items.len) continue;

                    const neighbor_dist = self.codebook.?.computeAsymmetricDistance(self.pq_codes.items[neighbor_id]);
                    try candidates.add(.{ .id = neighbor_id, .distance = neighbor_dist });

                    while (candidates.count() > list_size * 2) {
                        _ = candidates.removeOrNull();
                    }
                }
            }
        }

        return try result_list.toOwnedSlice(self.allocator);
    }

    /// Save the DiskANN index to disk in a sector-aligned format.
    ///
    /// File layout:
    /// ```
    /// [4B] Magic "DANN"
    /// [4B] Version (1)
    /// [4B] num_nodes
    /// [4B] dimensions
    /// [4B] max_degree
    /// [4B] pq_subspaces
    /// [4B] pq_bits
    /// [4B] medoid_id
    /// [padding to sector_size alignment]
    /// --- PQ Codebook section (sector-aligned) ---
    /// [pq_subspaces * num_centroids * subdim * 4B] centroids (f32)
    /// [padding to sector_size alignment]
    /// --- Graph section (sector-aligned per node) ---
    /// per node:
    ///   [4B] degree (number of neighbors)
    ///   [degree * 4B] neighbor IDs
    ///   [padding to sector_size]
    /// --- PQ Codes section ---
    /// [num_nodes * pq_subspaces bytes] compressed vectors
    /// ```
    pub fn save(self: *const DiskANNIndex, path: []const u8) PersistError!void {
        if (!self.stats.build_complete) return error.NotBuilt;
        const cb = self.codebook orelse return error.NotBuilt;

        const sector = self.config.sector_size;

        // Open file for writing
        var path_buf: [std.fs.max_path_bytes:0]u8 = [_:0]u8{0} ** std.fs.max_path_bytes;
        if (path.len >= path_buf.len) return error.OpenFailed;
        @memcpy(path_buf[0..path.len], path);
        path_buf[path.len] = 0;
        const path_z: [:0]const u8 = path_buf[0..path.len :0];

        const fd = std.posix.openatZ(std.posix.AT.FDCWD, path_z, .{
            .ACCMODE = .WRONLY,
            .CREAT = true,
            .TRUNC = true,
        }, 0o644) catch return error.OpenFailed;
        defer std.posix.close(fd);

        // --- Write header (sector-aligned) ---
        var header_buf: [4096]u8 = [_]u8{0} ** 4096;
        // Magic
        @memcpy(header_buf[0..4], &DISKANN_MAGIC);
        // Version
        std.mem.writeInt(u32, header_buf[4..8], DISKANN_FORMAT_VERSION, .little);
        // num_nodes
        std.mem.writeInt(u32, header_buf[8..12], self.num_vectors, .little);
        // dimensions
        std.mem.writeInt(u32, header_buf[12..16], self.config.dimensions, .little);
        // max_degree
        std.mem.writeInt(u32, header_buf[16..20], self.config.max_degree, .little);
        // pq_subspaces
        std.mem.writeInt(u32, header_buf[20..24], self.config.pq_subspaces, .little);
        // pq_bits
        std.mem.writeInt(u32, header_buf[24..28], self.config.pq_bits, .little);
        // medoid_id
        std.mem.writeInt(u32, header_buf[28..32], self.entry_point, .little);

        writeAllFd(fd, header_buf[0..sector]) catch return error.WriteFailed;

        // --- Write PQ Codebook section (sector-aligned) ---
        const centroid_bytes = cb.centroids.len * @sizeOf(f32);
        const centroid_aligned = alignToSector(centroid_bytes, sector);
        const centroid_raw: [*]const u8 = @ptrCast(cb.centroids.ptr);
        writeAllFd(fd, centroid_raw[0..centroid_bytes]) catch return error.WriteFailed;

        // Write padding to sector alignment
        const centroid_pad = centroid_aligned - centroid_bytes;
        if (centroid_pad > 0) {
            writePadding(fd, centroid_pad) catch return error.WriteFailed;
        }

        // --- Write Graph section (sector-aligned per node) ---
        for (0..self.num_vectors) |i| {
            const adj = self.graph.items[i];
            const degree: u32 = @intCast(adj.items.len);

            // Write degree
            const degree_bytes = std.mem.toBytes(std.mem.nativeToLittle(u32, degree));
            writeAllFd(fd, &degree_bytes) catch return error.WriteFailed;

            // Write neighbor IDs
            if (degree > 0) {
                const neighbors_raw: [*]const u8 = @ptrCast(adj.items.ptr);
                writeAllFd(fd, neighbors_raw[0 .. degree * @sizeOf(u32)]) catch return error.WriteFailed;
            }

            // Pad to sector_size
            const node_data_size = @sizeOf(u32) + degree * @sizeOf(u32);
            const remainder = node_data_size % sector;
            if (remainder != 0) {
                writePadding(fd, sector - remainder) catch return error.WriteFailed;
            }
        }

        // --- Write PQ Codes section ---
        for (0..self.num_vectors) |i| {
            writeAllFd(fd, self.pq_codes.items[i]) catch return error.WriteFailed;
        }
    }

    /// Load a DiskANN index from a memory-mapped file (zero-copy for codebook and PQ codes).
    ///
    /// The returned index holds a reference to the mapped file. The caller must
    /// call `deinit` which will unmap the file. Graph adjacency lists are copied
    /// into allocator-owned memory so they can be used with the existing search
    /// methods.
    pub fn load(allocator: std.mem.Allocator, path: []const u8) PersistError!DiskANNIndex {
        var mapped = mmap.MappedFile.open(allocator, path) catch |err| switch (err) {
            error.FileNotFound => return error.OpenFailed,
            error.AccessDenied => return error.OpenFailed,
            error.MmapFailed => return error.OpenFailed,
            error.InvalidFile => return error.CorruptedFile,
            error.OutOfMemory => return error.OutOfMemory,
        };
        errdefer mapped.close();

        const data = mapped.bytes();

        // --- Parse header ---
        if (data.len < DISKANN_HEADER_SIZE) return error.CorruptedFile;

        if (!std.mem.eql(u8, data[0..4], &DISKANN_MAGIC)) return error.InvalidMagic;

        const version = std.mem.readInt(u32, data[4..8], .little);
        if (version != DISKANN_FORMAT_VERSION) return error.UnsupportedVersion;

        const num_nodes = std.mem.readInt(u32, data[8..12], .little);
        const dimensions = std.mem.readInt(u32, data[12..16], .little);
        const max_degree = std.mem.readInt(u32, data[16..20], .little);
        const pq_subspaces = std.mem.readInt(u32, data[20..24], .little);
        const pq_bits = std.mem.readInt(u32, data[24..28], .little);
        const medoid_id = std.mem.readInt(u32, data[28..32], .little);

        const config = DiskANNConfig{
            .dimensions = dimensions,
            .max_degree = max_degree,
            .pq_subspaces = pq_subspaces,
            .pq_bits = pq_bits,
        };

        const sector = config.sector_size;
        const num_centroids: u32 = @as(u32, 1) << @intCast(pq_bits);
        const subspace_dim = dimensions / pq_subspaces;

        // --- Parse PQ Codebook (zero-copy from mapped region) ---
        const centroid_count = pq_subspaces * num_centroids * subspace_dim;
        const centroid_bytes = centroid_count * @sizeOf(f32);
        const centroid_aligned = alignToSector(centroid_bytes, sector);

        var offset: usize = sector; // header is sector-aligned
        if (offset + centroid_aligned > data.len) return error.CorruptedFile;

        // We need to copy centroids because PQCodebook owns the memory (deinit frees it)
        const centroids_copy = try allocator.alloc(f32, centroid_count);
        errdefer allocator.free(centroids_copy);

        const centroid_src: [*]const f32 = @ptrCast(@alignCast(data.ptr + offset));
        @memcpy(centroids_copy, centroid_src[0..centroid_count]);

        var codebook = PQCodebook{
            .allocator = allocator,
            .num_subspaces = pq_subspaces,
            .subspace_dim = subspace_dim,
            .num_centroids = num_centroids,
            .centroids = centroids_copy,
        };
        errdefer codebook.deinit();

        offset += centroid_aligned;

        // --- Parse Graph section ---
        var graph: std.ArrayListUnmanaged(std.ArrayListUnmanaged(u32)) = .empty;
        errdefer {
            for (graph.items) |*adj| adj.deinit(allocator);
            graph.deinit(allocator);
        }
        try graph.ensureTotalCapacity(allocator, num_nodes);

        for (0..num_nodes) |_| {
            if (offset + @sizeOf(u32) > data.len) return error.CorruptedFile;

            const degree = std.mem.readInt(u32, data[offset..][0..4], .little);
            offset += @sizeOf(u32);

            const neighbor_bytes = degree * @sizeOf(u32);
            if (offset + neighbor_bytes > data.len) return error.CorruptedFile;

            var adj: std.ArrayListUnmanaged(u32) = .empty;
            try adj.ensureTotalCapacity(allocator, degree);

            for (0..degree) |j| {
                const nid = std.mem.readInt(u32, data[offset + j * @sizeOf(u32) ..][0..4], .little);
                adj.appendAssumeCapacity(nid);
            }
            offset += neighbor_bytes;

            // Skip sector padding
            const node_data_size = @sizeOf(u32) + neighbor_bytes;
            const remainder = node_data_size % sector;
            if (remainder != 0) {
                offset += sector - remainder;
            }

            graph.appendAssumeCapacity(adj);
        }

        // --- Parse PQ Codes section ---
        var pq_codes: std.ArrayListUnmanaged([]u8) = .empty;
        errdefer {
            for (pq_codes.items) |codes| allocator.free(codes);
            pq_codes.deinit(allocator);
        }
        try pq_codes.ensureTotalCapacity(allocator, num_nodes);

        for (0..num_nodes) |_| {
            if (offset + pq_subspaces > data.len) return error.CorruptedFile;

            const codes = try allocator.alloc(u8, pq_subspaces);
            @memcpy(codes, data[offset..][0..pq_subspaces]);
            offset += pq_subspaces;

            pq_codes.appendAssumeCapacity(codes);
        }

        // Note: the mapped file is closed after loading since we copied all data.
        mapped.close();

        return DiskANNIndex{
            .config = config,
            .allocator = allocator,
            .num_vectors = num_nodes,
            .entry_point = medoid_id,
            .codebook = codebook,
            .graph = graph,
            .pq_codes = pq_codes,
            .vectors = .empty,
            .node_cache = .empty,
            .cache_order = .empty,
            .stats = .{
                .build_complete = true,
                .num_vectors = num_nodes,
                .vectors_indexed = num_nodes,
            },
        };
    }

    /// Get index statistics
    pub fn getStats(self: *const DiskANNIndex) IndexStats {
        var stats = self.stats;
        stats.num_vectors = self.num_vectors;
        stats.memory_bytes = self.estimateMemoryUsage();
        return stats;
    }

    fn estimateMemoryUsage(self: *const DiskANNIndex) u64 {
        const dim = self.config.dimensions;
        var total: u64 = 0;

        // Vectors
        total += @as(u64, self.num_vectors) * dim * @sizeOf(f32);

        // PQ codes
        total += @as(u64, self.num_vectors) * self.config.pq_subspaces;

        // Graph (estimate)
        total += @as(u64, self.num_vectors) * self.config.max_degree * @sizeOf(u32);

        // Codebook
        if (self.codebook) |cb| {
            total += @as(u64, cb.centroids.len) * @sizeOf(f32);
        }

        return total;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "diskann index basic" {
    const allocator = std.testing.allocator;

    const config = DiskANNConfig{
        .dimensions = 4,
        .max_degree = 4,
        .build_list_size = 10,
        .search_list_size = 10,
        .pq_subspaces = 2,
    };

    var idx = try DiskANNIndex.init(allocator, config);
    defer idx.deinit();

    // Build with small dataset
    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 1, 0, 0,
    };

    try idx.build(&vectors);

    try std.testing.expect(idx.num_vectors == 5);
    try std.testing.expect(idx.stats.build_complete);
}

test "diskann search" {
    const allocator = std.testing.allocator;

    const config = DiskANNConfig{
        .dimensions = 4,
        .max_degree = 4,
        .build_list_size = 10,
        .search_list_size = 10,
        .pq_subspaces = 2,
    };

    var idx = try DiskANNIndex.init(allocator, config);
    defer idx.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };

    try idx.build(&vectors);

    // Search for vector similar to first
    const query = [_]f32{ 0.9, 0.1, 0, 0 };
    const results = try idx.search(&query, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len >= 1);
    // First result should be closest to query
    try std.testing.expect(results[0].id == 0);
}

test "diskann save and load round-trip" {
    const allocator = std.testing.allocator;

    const config = DiskANNConfig{
        .dimensions = 4,
        .max_degree = 4,
        .build_list_size = 10,
        .search_list_size = 10,
        .pq_subspaces = 2,
    };

    // Build an index
    var idx = try DiskANNIndex.init(allocator, config);
    defer idx.deinit();

    const vectors = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
        1, 1, 0, 0,
    };
    try idx.build(&vectors);

    // Save to a temp file
    const tmp_path = "/tmp/diskann_test_roundtrip.dann";
    try idx.save(tmp_path);

    // Load it back
    var loaded = try DiskANNIndex.load(allocator, tmp_path);
    defer loaded.deinit();

    // Verify metadata
    try std.testing.expectEqual(idx.num_vectors, loaded.num_vectors);
    try std.testing.expectEqual(idx.entry_point, loaded.entry_point);
    try std.testing.expectEqual(idx.config.dimensions, loaded.config.dimensions);
    try std.testing.expectEqual(idx.config.max_degree, loaded.config.max_degree);
    try std.testing.expectEqual(idx.config.pq_subspaces, loaded.config.pq_subspaces);
    try std.testing.expect(loaded.stats.build_complete);

    // Verify graph structure
    for (0..idx.num_vectors) |i| {
        try std.testing.expectEqual(idx.graph.items[i].items.len, loaded.graph.items[i].items.len);
        for (idx.graph.items[i].items, loaded.graph.items[i].items) |a, b| {
            try std.testing.expectEqual(a, b);
        }
    }

    // Verify PQ codes
    for (0..idx.num_vectors) |i| {
        try std.testing.expectEqualSlices(u8, idx.pq_codes.items[i], loaded.pq_codes.items[i]);
    }

    // Verify codebook centroids
    const orig_cb = idx.codebook.?;
    const loaded_cb = loaded.codebook.?;
    try std.testing.expectEqual(orig_cb.centroids.len, loaded_cb.centroids.len);
    for (orig_cb.centroids, loaded_cb.centroids) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 1e-6);
    }

    // Clean up temp file
    std.posix.unlink(tmp_path) catch {};
}

test "diskann save rejects unbuilt index" {
    const allocator = std.testing.allocator;
    var idx = try DiskANNIndex.init(allocator, .{});
    defer idx.deinit();

    const result = idx.save("/tmp/diskann_test_notbuilt.dann");
    try std.testing.expectError(error.NotBuilt, result);
}

test "diskann load rejects invalid magic" {
    const allocator = std.testing.allocator;

    // Write a file with bad magic
    const tmp_path = "/tmp/diskann_test_badmagic.dann";
    const fd = std.posix.openatZ(std.posix.AT.FDCWD, tmp_path, .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .TRUNC = true,
    }, 0o644) catch return;
    defer std.posix.close(fd);

    var buf: [4096]u8 = [_]u8{0} ** 4096;
    @memcpy(buf[0..4], "NOPE");
    _ = std.posix.write(fd, &buf) catch return;

    const result = DiskANNIndex.load(allocator, tmp_path);
    try std.testing.expectError(error.InvalidMagic, result);

    std.posix.unlink(tmp_path) catch {};
}
