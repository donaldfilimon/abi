//! Mock Implementations for Integration Testing
//!
//! Provides mock implementations of hardware-dependent features that enable
//! integration tests to run on any system (including CI without GPUs).
//!
//! ## Design Principles
//!
//! - Mocks produce valid (not necessarily identical) results to real implementations
//! - All mocks are deterministic with fixed seeds
//! - Memory allocation follows real patterns for testing cleanup
//! - Error injection available for negative test cases

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const build_options = @import("build_options");

// ============================================================================
// Mock GPU Context
// ============================================================================

/// Mock GPU context for testing without real GPU hardware
pub const MockGpu = struct {
    allocator: std.mem.Allocator,
    device_buffers: std.ArrayListUnmanaged(DeviceBuffer),
    operation_count: u64,
    is_available: bool,

    /// Simulated device buffer
    pub const DeviceBuffer = struct {
        size: usize,
        data: []u8,
        id: u64,
    };

    pub fn init(allocator: std.mem.Allocator) MockGpu {
        return .{
            .allocator = allocator,
            .device_buffers = .{},
            .operation_count = 0,
            .is_available = true,
        };
    }

    pub fn deinit(self: *MockGpu) void {
        for (self.device_buffers.items) |buf| {
            self.allocator.free(buf.data);
        }
        self.device_buffers.deinit(self.allocator);
    }

    /// Simulate device memory allocation
    pub fn allocate(self: *MockGpu, size: usize) !*DeviceBuffer {
        const data = try self.allocator.alloc(u8, size);
        errdefer self.allocator.free(data);

        const buf = DeviceBuffer{
            .size = size,
            .data = data,
            .id = self.operation_count,
        };
        try self.device_buffers.append(self.allocator, buf);
        self.operation_count += 1;

        return &self.device_buffers.items[self.device_buffers.items.len - 1];
    }

    /// Simulate host-to-device copy
    pub fn copyToDevice(self: *MockGpu, buf: *DeviceBuffer, data: []const u8) void {
        _ = self;
        const copy_len = @min(buf.size, data.len);
        @memcpy(buf.data[0..copy_len], data[0..copy_len]);
    }

    /// Simulate device-to-host copy
    pub fn copyFromDevice(self: *MockGpu, buf: *DeviceBuffer, dest: []u8) void {
        _ = self;
        const copy_len = @min(buf.size, dest.len);
        @memcpy(dest[0..copy_len], buf.data[0..copy_len]);
    }

    /// Mock vector addition (CPU fallback)
    pub fn vectorAdd(self: *MockGpu, a: []const f32, b: []const f32, result: []f32) void {
        _ = self;
        for (a, b, result) |av, bv, *rv| {
            rv.* = av + bv;
        }
    }

    /// Mock matrix-vector multiply (CPU fallback)
    pub fn matrixVectorMul(
        self: *MockGpu,
        matrix: []const f32,
        vector: []const f32,
        result: []f32,
        rows: usize,
        cols: usize,
    ) void {
        _ = self;
        for (0..rows) |i| {
            var sum: f32 = 0;
            for (0..cols) |j| {
                sum += matrix[i * cols + j] * vector[j];
            }
            result[i] = sum;
        }
    }

    /// Check if mock GPU is "available"
    pub fn isAvailable(self: *const MockGpu) bool {
        return self.is_available;
    }

    /// Set availability (for testing error paths)
    pub fn setAvailable(self: *MockGpu, available: bool) void {
        self.is_available = available;
    }
};

// ============================================================================
// Mock LLM Model
// ============================================================================

/// Mock LLM model for testing without real model weights
pub const MockLlmModel = struct {
    allocator: std.mem.Allocator,
    vocab_size: u32,
    hidden_dim: u32,
    is_loaded: bool,
    generation_count: u64,

    /// Canned responses for testing
    const canned_responses = [_][]const u8{
        "This is a test response.",
        "The answer is 42.",
        "Hello, world!",
        "I am a mock language model.",
    };

    pub fn init(allocator: std.mem.Allocator) MockLlmModel {
        return .{
            .allocator = allocator,
            .vocab_size = 32000,
            .hidden_dim = 4096,
            .is_loaded = true,
            .generation_count = 0,
        };
    }

    pub fn deinit(self: *MockLlmModel) void {
        self.is_loaded = false;
    }

    /// Mock tokenization (simple byte-level)
    pub fn tokenize(self: *MockLlmModel, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        _ = self;
        const tokens = try allocator.alloc(u32, text.len);
        for (text, 0..) |c, i| {
            tokens[i] = @as(u32, c);
        }
        return tokens;
    }

    /// Mock detokenization
    pub fn detokenize(self: *MockLlmModel, tokens: []const u32, allocator: std.mem.Allocator) ![]u8 {
        _ = self;
        const text = try allocator.alloc(u8, tokens.len);
        for (tokens, 0..) |t, i| {
            text[i] = @truncate(t);
        }
        return text;
    }

    /// Mock text generation (returns canned response)
    pub fn generate(self: *MockLlmModel, prompt: []const u8, max_tokens: u32, allocator: std.mem.Allocator) ![]u8 {
        _ = prompt;
        _ = max_tokens;
        const idx = self.generation_count % canned_responses.len;
        self.generation_count += 1;
        return try allocator.dupe(u8, canned_responses[idx]);
    }

    /// Mock embedding generation
    pub fn embed(self: *MockLlmModel, text: []const u8, allocator: std.mem.Allocator) ![]f32 {
        _ = self;
        const embed_dim: usize = 384;
        const embeddings = try allocator.alloc(f32, embed_dim);

        // Generate deterministic pseudo-embeddings from text hash
        var hash: u64 = 0;
        for (text) |c| {
            hash = hash *% 31 +% @as(u64, c);
        }

        var rng = std.Random.DefaultPrng.init(hash);
        for (embeddings) |*e| {
            e.* = rng.random().float(f32) * 2.0 - 1.0;
        }

        return embeddings;
    }

    pub fn isLoaded(self: *const MockLlmModel) bool {
        return self.is_loaded;
    }
};

// ============================================================================
// Mock Network Registry
// ============================================================================

/// Mock network registry for testing without real sockets
pub const MockNetworkRegistry = struct {
    allocator: std.mem.Allocator,
    nodes: std.StringHashMapUnmanaged(NodeInfo),
    local_node_id: []const u8,

    pub const NodeInfo = struct {
        id: []const u8,
        address: []const u8,
        port: u16,
        is_healthy: bool,
        last_seen_ns: i64,
    };

    pub fn init(allocator: std.mem.Allocator) !MockNetworkRegistry {
        const local_id = try allocator.dupe(u8, "local-node-001");
        return .{
            .allocator = allocator,
            .nodes = .{},
            .local_node_id = local_id,
        };
    }

    pub fn deinit(self: *MockNetworkRegistry) void {
        var iter = self.nodes.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.id);
            self.allocator.free(entry.value_ptr.address);
        }
        self.nodes.deinit(self.allocator);
        self.allocator.free(self.local_node_id);
    }

    /// Register a node
    pub fn register(self: *MockNetworkRegistry, id: []const u8, address: []const u8, port: u16) !void {
        const id_copy = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(id_copy);

        const addr_copy = try self.allocator.dupe(u8, address);
        errdefer self.allocator.free(addr_copy);

        const key_copy = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(key_copy);

        try self.nodes.put(self.allocator, key_copy, .{
            .id = id_copy,
            .address = addr_copy,
            .port = port,
            .is_healthy = true,
            .last_seen_ns = 0,
        });
    }

    /// Unregister a node
    pub fn unregister(self: *MockNetworkRegistry, id: []const u8) bool {
        if (self.nodes.fetchRemove(id)) |entry| {
            self.allocator.free(entry.key);
            self.allocator.free(entry.value.id);
            self.allocator.free(entry.value.address);
            return true;
        }
        return false;
    }

    /// Get node count
    pub fn nodeCount(self: *const MockNetworkRegistry) usize {
        return self.nodes.count();
    }

    /// Check if node exists
    pub fn hasNode(self: *const MockNetworkRegistry, id: []const u8) bool {
        return self.nodes.contains(id);
    }
};

// ============================================================================
// Mock Replication Manager
// ============================================================================

/// Mock replication manager for HA testing
pub const MockReplicationManager = struct {
    allocator: std.mem.Allocator,
    replicas: std.ArrayListUnmanaged(ReplicaState),
    is_primary: bool,
    replication_lag_ns: i64,

    pub const ReplicaState = struct {
        id: []const u8,
        lsn: u64, // Log sequence number
        is_synced: bool,
    };

    pub fn init(allocator: std.mem.Allocator) MockReplicationManager {
        return .{
            .allocator = allocator,
            .replicas = .{},
            .is_primary = true,
            .replication_lag_ns = 0,
        };
    }

    pub fn deinit(self: *MockReplicationManager) void {
        for (self.replicas.items) |r| {
            self.allocator.free(r.id);
        }
        self.replicas.deinit(self.allocator);
    }

    /// Add a replica
    pub fn addReplica(self: *MockReplicationManager, id: []const u8) !void {
        const id_copy = try self.allocator.dupe(u8, id);
        try self.replicas.append(self.allocator, .{
            .id = id_copy,
            .lsn = 0,
            .is_synced = false,
        });
    }

    /// Simulate replication of a write
    pub fn replicate(self: *MockReplicationManager, lsn: u64) void {
        for (self.replicas.items) |*r| {
            r.lsn = lsn;
            r.is_synced = true;
        }
    }

    /// Get sync status
    pub fn allSynced(self: *const MockReplicationManager) bool {
        for (self.replicas.items) |r| {
            if (!r.is_synced) return false;
        }
        return true;
    }

    /// Simulate failover
    pub fn failover(self: *MockReplicationManager) void {
        self.is_primary = !self.is_primary;
    }
};

// ============================================================================
// Mock Database Context
// ============================================================================

/// Mock database for testing vector operations
pub const MockDatabase = struct {
    allocator: std.mem.Allocator,
    vectors: std.ArrayListUnmanaged(StoredVector),
    dimension: u32,

    pub const StoredVector = struct {
        id: u64,
        data: []f32,
        metadata: ?[]const u8,
    };

    pub fn init(allocator: std.mem.Allocator, dimension: u32) MockDatabase {
        return .{
            .allocator = allocator,
            .vectors = .{},
            .dimension = dimension,
        };
    }

    pub fn deinit(self: *MockDatabase) void {
        for (self.vectors.items) |v| {
            self.allocator.free(v.data);
            if (v.metadata) |m| self.allocator.free(m);
        }
        self.vectors.deinit(self.allocator);
    }

    /// Insert a vector
    pub fn insert(self: *MockDatabase, id: u64, data: []const f32, metadata: ?[]const u8) !void {
        const data_copy = try self.allocator.dupe(f32, data);
        errdefer self.allocator.free(data_copy);

        const meta_copy = if (metadata) |m| try self.allocator.dupe(u8, m) else null;
        errdefer if (meta_copy) |m| self.allocator.free(m);

        try self.vectors.append(self.allocator, .{
            .id = id,
            .data = data_copy,
            .metadata = meta_copy,
        });
    }

    /// Search for similar vectors (brute force cosine similarity)
    pub fn search(self: *MockDatabase, query: []const f32, k: usize, allocator: std.mem.Allocator) ![]SearchResult {
        const results = try allocator.alloc(SearchResult, @min(k, self.vectors.items.len));
        var result_count: usize = 0;

        // Compute all similarities
        var scores = try self.allocator.alloc(ScoreEntry, self.vectors.items.len);
        defer self.allocator.free(scores);

        for (self.vectors.items, 0..) |v, i| {
            scores[i] = .{
                .index = i,
                .score = cosineSimilarity(query, v.data),
            };
        }

        // Sort by score (descending)
        std.mem.sort(ScoreEntry, scores, {}, struct {
            fn lessThan(_: void, a: ScoreEntry, b: ScoreEntry) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Return top k
        for (scores[0..@min(k, scores.len)]) |s| {
            results[result_count] = .{
                .id = self.vectors.items[s.index].id,
                .score = s.score,
            };
            result_count += 1;
        }

        return results[0..result_count];
    }

    pub const SearchResult = struct {
        id: u64,
        score: f32,
    };

    const ScoreEntry = struct {
        index: usize,
        score: f32,
    };

    fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        var dot: f32 = 0;
        var norm_a: f32 = 0;
        var norm_b: f32 = 0;

        for (a, b) |av, bv| {
            dot += av * bv;
            norm_a += av * av;
            norm_b += bv * bv;
        }

        const denom = @sqrt(norm_a) * @sqrt(norm_b);
        return if (denom > 0) dot / denom else 0;
    }

    /// Get vector count
    pub fn count(self: *const MockDatabase) usize {
        return self.vectors.items.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "mock gpu operations" {
    var gpu = MockGpu.init(std.testing.allocator);
    defer gpu.deinit();

    try std.testing.expect(gpu.isAvailable());

    // Test vector add
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    var result: [3]f32 = undefined;
    gpu.vectorAdd(&a, &b, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 5), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9), result[2], 0.001);
}

test "mock llm operations" {
    var llm = MockLlmModel.init(std.testing.allocator);
    defer llm.deinit();

    try std.testing.expect(llm.isLoaded());

    // Test tokenization
    const tokens = try llm.tokenize("hello", std.testing.allocator);
    defer std.testing.allocator.free(tokens);
    try std.testing.expectEqual(@as(usize, 5), tokens.len);

    // Test generation
    const response = try llm.generate("test prompt", 100, std.testing.allocator);
    defer std.testing.allocator.free(response);
    try std.testing.expect(response.len > 0);
}

test "mock network registry" {
    var registry = try MockNetworkRegistry.init(std.testing.allocator);
    defer registry.deinit();

    try registry.register("node-1", "127.0.0.1", 8080);
    try registry.register("node-2", "127.0.0.2", 8081);

    try std.testing.expectEqual(@as(usize, 2), registry.nodeCount());
    try std.testing.expect(registry.hasNode("node-1"));

    _ = registry.unregister("node-1");
    try std.testing.expectEqual(@as(usize, 1), registry.nodeCount());
}

test "mock database search" {
    var db = MockDatabase.init(std.testing.allocator, 4);
    defer db.deinit();

    // Insert test vectors
    try db.insert(1, &[_]f32{ 1, 0, 0, 0 }, null);
    try db.insert(2, &[_]f32{ 0, 1, 0, 0 }, null);
    try db.insert(3, &[_]f32{ 0.9, 0.1, 0, 0 }, null);

    // Search
    const query = [_]f32{ 1, 0, 0, 0 };
    const results = try db.search(&query, 2, std.testing.allocator);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    // First result should be exact match (id=1)
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

test "mock replication manager" {
    var rep = MockReplicationManager.init(std.testing.allocator);
    defer rep.deinit();

    try rep.addReplica("replica-1");
    try rep.addReplica("replica-2");

    try std.testing.expect(!rep.allSynced());

    rep.replicate(100);
    try std.testing.expect(rep.allSynced());
}
