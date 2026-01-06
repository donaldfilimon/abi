//! Vector index implementations for the database module.
const std = @import("std");

const simd = @import("../../shared/simd.zig");
const binary = @import("../../shared/utils/binary.zig");

const index_magic = "ABIX";
const index_version: u16 = 1;

pub const IndexResult = struct {
    id: u64,
    score: f32,
};

pub const VectorRecordView = struct {
    id: u64,
    vector: []const f32,
};

pub const IndexType = enum {
    none,
    hnsw,
    ivf_pq,
};

pub const IndexConfig = struct {
    index_type: IndexType = .hnsw,
    auto_rebuild: bool = true,
    min_records: usize = 32,
    hnsw_m: usize = 16,
    ivf_clusters: usize = 8,
    ivf_probe: usize = 2,
    pq_bits: u8 = 8,
};

pub const IndexError = error{
    InvalidConfiguration,
    EmptyIndex,
    InvalidData,
};

pub const SaveError =
    std.Io.File.OpenError ||
    std.Io.File.Writer.Error ||
    std.mem.Allocator.Error ||
    IndexError;
pub const LoadError =
    std.Io.Dir.ReadFileAllocError ||
    std.mem.Allocator.Error ||
    IndexError;

pub const IndexManager = struct {
    config: IndexConfig,
    dirty: bool,
    index: ?VectorIndex,
    record_count: usize,

    /// Initialize an index manager for the provided config.
    pub fn init(config: IndexConfig) IndexManager {
        return .{
            .config = config,
            .dirty = true,
            .index = null,
            .record_count = 0,
        };
    }

    /// Release any index data held by the manager.
    pub fn deinit(self: *IndexManager, allocator: std.mem.Allocator) void {
        if (self.index) |*index| {
            index.deinit(allocator);
        }
        self.* = undefined;
    }

    /// Mark the index as dirty after data mutations.
    pub fn markDirty(self: *IndexManager) void {
        self.dirty = true;
    }

    /// Update the index config and mark dirty.
    pub fn setConfig(self: *IndexManager, config: IndexConfig) void {
        self.config = config;
        self.dirty = true;
    }

    /// Build the index if configured and needed.
    pub fn buildIfNeeded(
        self: *IndexManager,
        allocator: std.mem.Allocator,
        records: []const VectorRecordView,
    ) !void {
        if (self.config.index_type == .none) return;
        if (records.len < self.config.min_records) return;
        if (!self.config.auto_rebuild and self.index != null) return;
        if (!self.dirty and self.index != null and self.record_count == records.len) {
            return;
        }
        if (self.index) |*index| {
            index.deinit(allocator);
            self.index = null;
        }

        self.index = switch (self.config.index_type) {
            .none => null,
            .hnsw => .{
                .hnsw = try HnswIndex.build(allocator, records, self.config.hnsw_m),
            },
            .ivf_pq => .{
                .ivf_pq = try IvfPqIndex.build(allocator, records, .{
                    .clusters = self.config.ivf_clusters,
                    .probe = self.config.ivf_probe,
                    .pq_bits = self.config.pq_bits,
                }),
            },
        };
        self.dirty = false;
        self.record_count = records.len;
    }

    /// Search using the index when available, falling back to brute force.
    pub fn search(
        self: *IndexManager,
        allocator: std.mem.Allocator,
        records: []const VectorRecordView,
        query: []const f32,
        top_k: usize,
    ) ![]IndexResult {
        if (records.len == 0) {
            return allocator.alloc(IndexResult, 0);
        }
        if (records[0].vector.len != query.len) {
            return allocator.alloc(IndexResult, 0);
        }

        if (self.config.index_type == .none or records.len < self.config.min_records) {
            return bruteForceSearch(allocator, records, query, top_k);
        }

        try self.buildIfNeeded(allocator, records);
        if (self.index) |*index| {
            return index.search(allocator, records, query, top_k);
        }

        return bruteForceSearch(allocator, records, query, top_k);
    }

    /// Persist the current index to a file.
    pub fn saveToFile(
        self: *const IndexManager,
        allocator: std.mem.Allocator,
        path: []const u8,
    ) SaveError!void {
        var writer = binary.SerializationWriter.init(allocator);
        defer writer.deinit();

        try writer.appendBytes(index_magic);
        try writer.appendInt(u16, index_version);
        try writer.appendInt(u8, @intFromEnum(self.config.index_type));
        try writer.appendInt(u32, @intCast(self.config.min_records));
        try writer.appendInt(u32, @intCast(self.config.hnsw_m));
        try writer.appendInt(u32, @intCast(self.config.ivf_clusters));
        try writer.appendInt(u32, @intCast(self.config.ivf_probe));
        try writer.appendInt(u8, self.config.pq_bits);
        try writer.appendInt(u64, @intCast(self.record_count));

        if (self.index) |index| {
            try index.save(&writer);
        }

        const bytes = try writer.toOwnedSlice();
        defer allocator.free(bytes);

        var io_backend = std.Io.Threaded.init(allocator, .{});
        defer io_backend.deinit();
        const io = io_backend.io();

        var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, bytes);
    }

    /// Load an index from a file and replace any existing index.
    pub fn loadFromFile(
        self: *IndexManager,
        allocator: std.mem.Allocator,
        path: []const u8,
    ) LoadError!void {
        var io_backend = std.Io.Threaded.init(allocator, .{});
        defer io_backend.deinit();
        const io = io_backend.io();

        const data = try std.Io.Dir.cwd().readFileAlloc(
            io,
            path,
            allocator,
            .limited(128 * 1024 * 1024),
        );
        defer allocator.free(data);

        var cursor = binary.SerializationCursor.init(data);
        const magic = try cursor.readBytes(index_magic.len);
        if (!std.mem.eql(u8, magic, index_magic)) return IndexError.InvalidData;

        const version = try cursor.readInt(u16);
        if (version != index_version) return IndexError.InvalidData;

        const index_type: IndexType = @enumFromInt(try cursor.readInt(u8));
        const min_records = try cursor.readInt(u32);
        const hnsw_m = try cursor.readInt(u32);
        const ivf_clusters = try cursor.readInt(u32);
        const ivf_probe = try cursor.readInt(u32);
        const pq_bits = try cursor.readInt(u8);
        const record_count = try cursor.readInt(u64);

        if (self.index) |*existing| {
            existing.deinit(allocator);
        }
        self.index = null;

        self.config = .{
            .index_type = index_type,
            .auto_rebuild = true,
            .min_records = min_records,
            .hnsw_m = hnsw_m,
            .ivf_clusters = ivf_clusters,
            .ivf_probe = ivf_probe,
            .pq_bits = pq_bits,
        };
        self.record_count = @intCast(record_count);
        self.dirty = false;

        switch (index_type) {
            .none => {},
            .hnsw => {
                const loaded = try HnswIndex.load(&cursor, allocator);
                self.index = .{ .hnsw = loaded };
            },
            .ivf_pq => {
                const loaded = try IvfPqIndex.load(&cursor, allocator);
                self.index = .{ .ivf_pq = loaded };
            },
        }
    }
};

pub const VectorIndex = union(enum) {
    hnsw: HnswIndex,
    ivf_pq: IvfPqIndex,

    pub fn deinit(self: *VectorIndex, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .hnsw => |*index| index.deinit(allocator),
            .ivf_pq => |*index| index.deinit(allocator),
        }
        self.* = undefined;
    }

    pub fn search(
        self: *VectorIndex,
        allocator: std.mem.Allocator,
        records: []const VectorRecordView,
        query: []const f32,
        top_k: usize,
    ) ![]IndexResult {
        return switch (self.*) {
            .hnsw => |*index| index.search(allocator, records, query, top_k),
            .ivf_pq => |*index| index.search(allocator, records, query, top_k),
        };
    }

    pub fn save(self: VectorIndex, writer: *binary.SerializationWriter) !void {
        switch (self) {
            .hnsw => |index| try index.save(writer),
            .ivf_pq => |index| try index.save(writer),
        }
    }
};

pub const HnswIndex = struct {
    m: usize,
    neighbors: []NeighborList,

    pub fn build(
        allocator: std.mem.Allocator,
        records: []const VectorRecordView,
        m: usize,
    ) !HnswIndex {
        if (records.len == 0) return IndexError.EmptyIndex;
        if (m == 0) return IndexError.InvalidConfiguration;

        const node_count = records.len;
        const neighbor_cap = @min(m, node_count - 1);
        var neighbors = try allocator.alloc(NeighborList, node_count);
        errdefer {
            for (neighbors) |list| allocator.free(list.nodes);
            allocator.free(neighbors);
        }

        var best_scores = try allocator.alloc(f32, neighbor_cap);
        defer allocator.free(best_scores);
        var best_indices = try allocator.alloc(u32, neighbor_cap);
        defer allocator.free(best_indices);

        for (records, 0..) |record, i| {
            @memset(best_scores, -std.math.inf(f32));
            @memset(best_indices, 0);
            for (records, 0..) |candidate, j| {
                if (i == j) continue;
                const score = simd.cosineSimilarity(record.vector, candidate.vector);
                var min_index: usize = 0;
                var min_score: f32 = best_scores[0];
                var k: usize = 1;
                while (k < best_scores.len) : (k += 1) {
                    if (best_scores[k] < min_score) {
                        min_score = best_scores[k];
                        min_index = k;
                    }
                }
                if (score > min_score) {
                    best_scores[min_index] = score;
                    best_indices[min_index] = @intCast(j);
                }
            }

            var count: usize = 0;
            var k: usize = 0;
            while (k < best_scores.len) : (k += 1) {
                if (best_scores[k] > -std.math.inf(f32)) {
                    count += 1;
                }
            }
            const list = try allocator.alloc(u32, count);
            var out_index: usize = 0;
            k = 0;
            while (k < best_scores.len) : (k += 1) {
                if (best_scores[k] > -std.math.inf(f32)) {
                    list[out_index] = best_indices[k];
                    out_index += 1;
                }
            }
            neighbors[i] = .{ .nodes = list };
        }

        return .{
            .m = neighbor_cap,
            .neighbors = neighbors,
        };
    }

    pub fn deinit(self: *HnswIndex, allocator: std.mem.Allocator) void {
        for (self.neighbors) |list| {
            allocator.free(list.nodes);
        }
        allocator.free(self.neighbors);
        self.* = undefined;
    }

    pub fn search(
        self: *HnswIndex,
        allocator: std.mem.Allocator,
        records: []const VectorRecordView,
        query: []const f32,
        top_k: usize,
    ) ![]IndexResult {
        if (records.len == 0) return allocator.alloc(IndexResult, 0);
        const entry: u32 = 0;

        var candidates = std.ArrayListUnmanaged(u32).empty;
        defer candidates.deinit(allocator);
        var seen = std.AutoHashMap(u32, void).init(allocator);
        defer seen.deinit();

        try addCandidate(&candidates, &seen, entry);
        if (entry < self.neighbors.len) {
            for (self.neighbors[entry].nodes) |neighbor| {
                try addCandidate(&candidates, &seen, neighbor);
            }
        }
        for (candidates.items) |candidate| {
            if (candidate >= self.neighbors.len) continue;
            for (self.neighbors[candidate].nodes) |neighbor| {
                try addCandidate(&candidates, &seen, neighbor);
            }
        }

        if (candidates.items.len == 0) {
            return bruteForceSearch(allocator, records, query, top_k);
        }

        var results = std.ArrayListUnmanaged(IndexResult).empty;
        errdefer results.deinit(allocator);
        try results.ensureTotalCapacity(allocator, candidates.items.len);
        for (candidates.items) |index| {
            if (index >= records.len) continue;
            const score = simd.cosineSimilarity(query, records[index].vector);
            results.appendAssumeCapacity(.{
                .id = records[index].id,
                .score = score,
            });
        }

        sortResults(results.items);
        if (top_k < results.items.len) {
            results.shrinkRetainingCapacity(top_k);
        }
        return results.toOwnedSlice(allocator);
    }

    pub fn save(self: HnswIndex, writer: *binary.SerializationWriter) !void {
        try writer.appendInt(u32, @intCast(self.neighbors.len));
        try writer.appendInt(u32, @intCast(self.m));
        for (self.neighbors) |list| {
            try writer.appendInt(u32, @intCast(list.nodes.len));
            for (list.nodes) |node| {
                try writer.appendInt(u32, node);
            }
        }
    }

    pub fn load(
        cursor: *binary.SerializationCursor,
        allocator: std.mem.Allocator,
    ) !HnswIndex {
        const node_count = try cursor.readInt(u32);
        const m = try cursor.readInt(u32);
        var neighbors = try allocator.alloc(NeighborList, node_count);
        errdefer allocator.free(neighbors);
        var i: usize = 0;
        while (i < node_count) : (i += 1) {
            const count = try cursor.readInt(u32);
            const list = try allocator.alloc(u32, count);
            var j: usize = 0;
            while (j < count) : (j += 1) {
                list[j] = try cursor.readInt(u32);
            }
            neighbors[i] = .{ .nodes = list };
        }
        return .{
            .m = @intCast(m),
            .neighbors = neighbors,
        };
    }
};

pub const IvfPqIndex = struct {
    dim: usize,
    clusters: []Cluster,
    assignments: []u32,
    codes: []u8,
    min_values: []f32,
    max_values: []f32,
    probes: usize,
    bits: u8,

    pub const BuildOptions = struct {
        clusters: usize,
        probe: usize,
        pq_bits: u8,
    };

    pub fn build(
        allocator: std.mem.Allocator,
        records: []const VectorRecordView,
        options: BuildOptions,
    ) !IvfPqIndex {
        if (records.len == 0) return IndexError.EmptyIndex;
        const dim = records[0].vector.len;
        if (dim == 0) return IndexError.InvalidData;

        const cluster_count = @max(@min(options.clusters, records.len), 1);
        var clusters = try allocator.alloc(Cluster, cluster_count);
        errdefer allocator.free(clusters);
        for (clusters, 0..) |*cluster, i| {
            const centroid = try allocator.alloc(f32, dim);
            std.mem.copyForwards(f32, centroid, records[i % records.len].vector);
            cluster.* = .{
                .centroid = centroid,
                .members = &.{},
            };
        }

        var assignments = try allocator.alloc(u32, records.len);
        errdefer allocator.free(assignments);

        var iter: usize = 0;
        while (iter < 3) : (iter += 1) {
            var counts = try allocator.alloc(usize, cluster_count);
            defer allocator.free(counts);
            @memset(counts, 0);
            var sums = try allocator.alloc(f32, cluster_count * dim);
            defer allocator.free(sums);
            @memset(sums, 0);

            for (records, 0..) |record, i| {
                const cluster_index = findNearestCentroid(record.vector, clusters);
                assignments[i] = @intCast(cluster_index);
                counts[cluster_index] += 1;
                const offset = cluster_index * dim;
                for (record.vector, 0..) |value, j| {
                    sums[offset + j] += value;
                }
            }

            for (clusters, 0..) |*cluster, i| {
                if (counts[i] == 0) continue;
                const offset = i * dim;
                for (cluster.centroid, 0..) |*value, j| {
                    value.* = sums[offset + j] / @as(f32, @floatFromInt(counts[i]));
                }
            }
        }

        const member_counts = try allocator.alloc(usize, cluster_count);
        defer allocator.free(member_counts);
        @memset(member_counts, 0);
        for (assignments) |cluster_id| {
            member_counts[cluster_id] += 1;
        }

        for (clusters, 0..) |*cluster, i| {
            cluster.members = try allocator.alloc(u32, member_counts[i]);
        }

        var offsets = try allocator.alloc(usize, cluster_count);
        defer allocator.free(offsets);
        @memset(offsets, 0);
        for (assignments, 0..) |cluster_id, i| {
            const idx = offsets[cluster_id];
            clusters[cluster_id].members[idx] = @intCast(i);
            offsets[cluster_id] += 1;
        }

        const min_values = try allocator.alloc(f32, dim);
        const max_values = try allocator.alloc(f32, dim);
        errdefer allocator.free(min_values);
        errdefer allocator.free(max_values);
        initializeMinMax(min_values, max_values, records);

        const codes = try allocator.alloc(u8, records.len * dim);
        errdefer allocator.free(codes);
        const bits = clampBits(options.pq_bits);
        encodeVectors(records, min_values, max_values, bits, codes);

        return .{
            .dim = dim,
            .clusters = clusters,
            .assignments = assignments,
            .codes = codes,
            .min_values = min_values,
            .max_values = max_values,
            .probes = @max(@min(options.probe, cluster_count), 1),
            .bits = bits,
        };
    }

    pub fn deinit(self: *IvfPqIndex, allocator: std.mem.Allocator) void {
        for (self.clusters) |cluster| {
            allocator.free(cluster.centroid);
            allocator.free(cluster.members);
        }
        allocator.free(self.clusters);
        allocator.free(self.assignments);
        allocator.free(self.codes);
        allocator.free(self.min_values);
        allocator.free(self.max_values);
        self.* = undefined;
    }

    pub fn search(
        self: *IvfPqIndex,
        allocator: std.mem.Allocator,
        records: []const VectorRecordView,
        query: []const f32,
        top_k: usize,
    ) ![]IndexResult {
        if (records.len == 0) return allocator.alloc(IndexResult, 0);
        if (query.len != self.dim) return IndexError.InvalidData;

        const probe_clusters = try selectTopClusters(
            allocator,
            self.clusters,
            query,
            self.probes,
        );
        defer allocator.free(probe_clusters);

        var candidates = std.ArrayListUnmanaged(u32).empty;
        defer candidates.deinit(allocator);
        for (probe_clusters) |cluster_id| {
            const members = self.clusters[cluster_id].members;
            try candidates.appendSlice(allocator, members);
        }

        if (candidates.items.len == 0) {
            return bruteForceSearch(allocator, records, query, top_k);
        }

        const decoded = try allocator.alloc(f32, self.dim);
        defer allocator.free(decoded);

        var results = std.ArrayListUnmanaged(IndexResult).empty;
        errdefer results.deinit(allocator);
        try results.ensureTotalCapacity(allocator, candidates.items.len);
        for (candidates.items) |index| {
            if (index >= records.len) continue;
            decodeVector(self, index, decoded);
            const score = simd.cosineSimilarity(query, decoded);
            results.appendAssumeCapacity(.{
                .id = records[index].id,
                .score = score,
            });
        }

        sortResults(results.items);
        if (top_k < results.items.len) {
            results.shrinkRetainingCapacity(top_k);
        }
        return results.toOwnedSlice(allocator);
    }

    pub fn save(self: IvfPqIndex, writer: *binary.SerializationWriter) !void {
        try writer.appendInt(u32, @intCast(self.dim));
        try writer.appendInt(u32, @intCast(self.clusters.len));
        try writer.appendInt(u32, @intCast(self.probes));
        try writer.appendInt(u8, self.bits);
        for (self.min_values) |value| {
            try writer.appendInt(u32, @bitCast(value));
        }
        for (self.max_values) |value| {
            try writer.appendInt(u32, @bitCast(value));
        }
        for (self.clusters) |cluster| {
            for (cluster.centroid) |value| {
                try writer.appendInt(u32, @bitCast(value));
            }
        }
        for (self.clusters) |cluster| {
            try writer.appendInt(u32, @intCast(cluster.members.len));
            for (cluster.members) |member| {
                try writer.appendInt(u32, member);
            }
        }
        try writer.appendInt(u32, @intCast(self.codes.len));
        try writer.appendBytes(self.codes);
    }

    pub fn load(
        cursor: *binary.SerializationCursor,
        allocator: std.mem.Allocator,
    ) !IvfPqIndex {
        const dim = try cursor.readInt(u32);
        const cluster_count = try cursor.readInt(u32);
        const probes = try cursor.readInt(u32);
        const bits = try cursor.readInt(u8);

        const min_values = try allocator.alloc(f32, dim);
        errdefer allocator.free(min_values);
        const max_values = try allocator.alloc(f32, dim);
        errdefer allocator.free(max_values);
        var i: usize = 0;
        while (i < dim) : (i += 1) {
            min_values[i] = @bitCast(try cursor.readInt(u32));
        }
        i = 0;
        while (i < dim) : (i += 1) {
            max_values[i] = @bitCast(try cursor.readInt(u32));
        }

        const clusters = try allocator.alloc(Cluster, cluster_count);
        errdefer allocator.free(clusters);
        for (clusters) |*cluster| {
            const centroid = try allocator.alloc(f32, dim);
            for (centroid, 0..) |*value, _| {
                value.* = @bitCast(try cursor.readInt(u32));
            }
            cluster.* = .{
                .centroid = centroid,
                .members = &.{},
            };
        }

        for (clusters) |*cluster| {
            const count = try cursor.readInt(u32);
            const members = try allocator.alloc(u32, count);
 for (members, 0..) |*member, i| {
                member.* = try cursor.readInt(u32);
            }
            cluster.members = members;
        }

        const codes_len = try cursor.readInt(u32);
        const codes = try allocator.alloc(u8, codes_len);
        const raw = try cursor.readBytes(codes_len);
        std.mem.copyForwards(u8, codes, raw);

        return .{
            .dim = dim,
            .clusters = clusters,
            .assignments = &.{},
            .codes = codes,
            .min_values = min_values,
            .max_values = max_values,
            .probes = probes,
            .bits = bits,
        };
    }
};

const NeighborList = struct {
    nodes: []u32,
};

pub const Cluster = struct {
    centroid: []f32,
    members: []u32,
};

fn bruteForceSearch(
    allocator: std.mem.Allocator,
    records: []const VectorRecordView,
    query: []const f32,
    top_k: usize,
) ![]IndexResult {
    var results = std.ArrayListUnmanaged(IndexResult).empty;
    errdefer results.deinit(allocator);
    try results.ensureTotalCapacity(allocator, records.len);
    for (records) |record| {
        const score = simd.cosineSimilarity(query, record.vector);
        results.appendAssumeCapacity(.{ .id = record.id, .score = score });
    }
    sortResults(results.items);
    if (top_k < results.items.len) {
        results.shrinkRetainingCapacity(top_k);
    }
    return results.toOwnedSlice(allocator);
}

fn sortResults(results: []IndexResult) void {
    std.sort.pdq(IndexResult, results, {}, struct {
        fn lessThan(_: void, lhs: IndexResult, rhs: IndexResult) bool {
            return lhs.score > rhs.score;
        }
    }.lessThan);
}

fn addCandidate(
    list: *std.ArrayListUnmanaged(u32),
    seen: *std.AutoHashMap(u32, void),
    index: u32,
) !void {
    if (seen.contains(index)) return;
    try seen.put(index, {});
    try list.append(seen.allocator, index);
}

fn findNearestCentroid(vector: []const f32, clusters: []const Cluster) usize {
    var best_index: usize = 0;
    var best_score: f32 = -std.math.inf(f32);
    for (clusters, 0..) |cluster, i| {
        const score = simd.cosineSimilarity(vector, cluster.centroid);
        if (score > best_score) {
            best_score = score;
            best_index = i;
        }
    }
    return best_index;
}

fn initializeMinMax(
    min_values: []f32,
    max_values: []f32,
    records: []const VectorRecordView,
) void {
    @memset(min_values, std.math.inf(f32));
    @memset(max_values, -std.math.inf(f32));
    for (records) |record| {
        for (record.vector, 0..) |value, i| {
            if (value < min_values[i]) min_values[i] = value;
            if (value > max_values[i]) max_values[i] = value;
        }
    }
}

fn clampBits(bits: u8) u8 {
    if (bits == 0) return 8;
    return @min(bits, 8);
}

fn encodeVectors(
    records: []const VectorRecordView,
    min_values: []const f32,
    max_values: []const f32,
    bits: u8,
    codes: []u8,
) void {
    const levels = (@as(u16, 1) << bits) - 1;
    const dim = min_values.len;
    for (records, 0..) |record, i| {
        const offset = i * dim;
        for (record.vector, 0..) |value, j| {
            const min_val = min_values[j];
            const max_val = max_values[j];
            const code = quantizeValue(value, min_val, max_val, levels);
            codes[offset + j] = code;
        }
    }
}

fn quantizeValue(value: f32, min_val: f32, max_val: f32, levels: u16) u8 {
    if (max_val <= min_val) return 0;
    const normalized = (value - min_val) / (max_val - min_val);
    const scaled = normalized * @as(f32, @floatFromInt(levels));
    const clamped = std.math.clamp(scaled, 0.0, @as(f32, @floatFromInt(levels)));
    return @intCast(@as(u16, @intFromFloat(clamped)));
}

fn decodeVector(index: *const IvfPqIndex, record_index: u32, out: []f32) void {
    const levels = (@as(u16, 1) << index.bits) - 1;
    const offset = @as(usize, record_index) * index.dim;
    for (out, 0..) |*value, i| {
        const code = index.codes[offset + i];
        value.* = decodeValue(
            code,
            index.min_values[i],
            index.max_values[i],
            levels,
        );
    }
}

fn decodeValue(code: u8, min_val: f32, max_val: f32, levels: u16) f32 {
    if (max_val <= min_val or levels == 0) return min_val;
    const fraction = @as(f32, @floatFromInt(code)) / @as(f32, @floatFromInt(levels));
    return min_val + fraction * (max_val - min_val);
}

fn selectTopClusters(
    allocator: std.mem.Allocator,
    clusters: []const Cluster,
    query: []const f32,
    probe: usize,
) ![]usize {
    const count = @min(probe, clusters.len);
    var best_scores = try allocator.alloc(f32, count);
    defer allocator.free(best_scores);
    var best_indices = try allocator.alloc(usize, count);
    defer allocator.free(best_indices);
    @memset(best_scores, -std.math.inf(f32));
    @memset(best_indices, 0);

    for (clusters, 0..) |cluster, i| {
        const score = simd.cosineSimilarity(query, cluster.centroid);
        var min_index: usize = 0;
        var min_score: f32 = best_scores[0];
        var j: usize = 1;
        while (j < best_scores.len) : (j += 1) {
            if (best_scores[j] < min_score) {
                min_score = best_scores[j];
                min_index = j;
            }
        }
        if (score > min_score) {
            best_scores[min_index] = score;
            best_indices[min_index] = i;
        }
    }

    const result = try allocator.alloc(usize, count);
    std.mem.copyForwards(usize, result, best_indices);
    return result;
}

test "hnsw index builds and searches" {
    const records = [_]VectorRecordView{
        .{ .id = 1, .vector = &.{ 1.0, 0.0 } },
        .{ .id = 2, .vector = &.{ 0.0, 1.0 } },
        .{ .id = 3, .vector = &.{ 1.0, 1.0 } },
    };
    var index = try HnswIndex.build(std.testing.allocator, &records, 2);
    defer index.deinit(std.testing.allocator);

    const results = try index.search(std.testing.allocator, &records, &.{ 1.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);
    try std.testing.expect(results.len > 0);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

test "ivf-pq index builds and searches" {
    const records = [_]VectorRecordView{
        .{ .id = 1, .vector = &.{ 1.0, 0.0 } },
        .{ .id = 2, .vector = &.{ 0.0, 1.0 } },
        .{ .id = 3, .vector = &.{ 1.0, 1.0 } },
        .{ .id = 4, .vector = &.{ 0.9, 0.1 } },
    };
    var index = try IvfPqIndex.build(std.testing.allocator, &records, .{
        .clusters = 2,
        .probe = 1,
        .pq_bits = 8,
    });
    defer index.deinit(std.testing.allocator);

    const results = try index.search(std.testing.allocator, &records, &.{ 1.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);
    try std.testing.expect(results.len > 0);
    try std.testing.expect(results[0].score >= results[results.len - 1].score);
}

test "index manager save and load" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const file = try tmp.dir.createFile(io, "index.bin", .{ .truncate = true });
    file.close(io);
    const path_z = try tmp.dir.realPathFileAlloc(io, "index.bin", allocator);
    defer allocator.free(path_z);
    const path = path_z[0..path_z.len];

    const records = [_]VectorRecordView{
        .{ .id = 1, .vector = &.{ 1.0, 0.0 } },
        .{ .id = 2, .vector = &.{ 0.0, 1.0 } },
        .{ .id = 3, .vector = &.{ 1.0, 1.0 } },
    };

    var manager = IndexManager.init(.{ .index_type = .hnsw, .min_records = 1 });
    defer manager.deinit(allocator);
    try manager.buildIfNeeded(allocator, &records);
    try manager.saveToFile(allocator, path);

    var loaded = IndexManager.init(.{ .index_type = .none });
    defer loaded.deinit(allocator);
    try loaded.loadFromFile(allocator, path);

    const results = try loaded.search(allocator, &records, &.{ 1.0, 0.0 }, 2);
    defer allocator.free(results);
    try std.testing.expect(results.len > 0);
}
