//! Internal graph structures for parallel HNSW construction.
//!
//! Contains the thread-safe parallel graph, per-node state, and per-worker
//! insertion logic. Used by ParallelHnswBuilder in parallel_hnsw.zig.

const std = @import("std");
const hnsw = @import("hnsw.zig");
const index_mod = @import("index.zig");
const simd = @import("../../services/shared/simd.zig");

// ============================================================================
// Insert Task
// ============================================================================

/// Task for work-stealing scheduler.
pub const InsertTask = struct {
    node_id: u32,
    priority: u8, // Higher layer = higher priority
};

// ============================================================================
// Parallel Graph Structure
// ============================================================================

/// Node state during parallel construction.
pub const ParallelNodeState = struct {
    /// Target layer for this node
    target_layer: i32,
    /// Insertion completed flag
    inserted: std.atomic.Value(bool),
    /// Lock for neighbor list updates
    mutex: std.Thread.Mutex,
};

/// Thread-safe parallel HNSW graph for concurrent construction.
pub const ParallelGraph = struct {
    /// Per-node state
    node_states: []ParallelNodeState,
    /// Neighbor lists per node per layer (indexed: [node][layer])
    neighbors: [][]std.ArrayListUnmanaged(u32),
    /// Current entry point (atomic for concurrent updates)
    entry_point: std.atomic.Value(u32),
    /// Current max layer (atomic)
    max_layer: std.atomic.Value(i32),
    /// Graph parameters
    m: usize,
    m_max: usize,
    m_max0: usize,
    /// Allocator
    allocator: std.mem.Allocator,
    /// Statistics
    stats: struct {
        lock_contentions: std.atomic.Value(u64),
        distance_computations: std.atomic.Value(u64),
    },

    pub const INVALID_ENTRY: u32 = std.math.maxInt(u32);

    pub fn init(allocator: std.mem.Allocator, node_count: usize, m: usize, layer_assignments: []const i32) !ParallelGraph {
        const node_states = try allocator.alloc(ParallelNodeState, node_count);
        errdefer allocator.free(node_states);

        const neighbors = try allocator.alloc([]std.ArrayListUnmanaged(u32), node_count);
        // Track how many nodes we've initialized for cleanup on error
        var initialized_nodes: usize = 0;
        errdefer {
            for (neighbors[0..initialized_nodes]) |node_neighbors| {
                for (node_neighbors) |*list| {
                    list.deinit(allocator);
                }
                allocator.free(node_neighbors);
            }
            allocator.free(neighbors);
        }

        var max_layer_val: i32 = 0;

        for (node_states, neighbors, layer_assignments) |*state, *node_neighbors, layer| {
            state.* = .{
                .target_layer = layer,
                .inserted = std.atomic.Value(bool).init(false),
                .mutex = .{},
            };

            const layer_count: usize = @intCast(layer + 1);
            node_neighbors.* = try allocator.alloc(std.ArrayListUnmanaged(u32), layer_count);
            for (node_neighbors.*) |*list| {
                list.* = .empty;
            }

            if (layer > max_layer_val) {
                max_layer_val = layer;
            }

            initialized_nodes += 1;
        }

        return .{
            .node_states = node_states,
            .neighbors = neighbors,
            .entry_point = std.atomic.Value(u32).init(INVALID_ENTRY),
            .max_layer = std.atomic.Value(i32).init(max_layer_val),
            .m = m,
            .m_max = m,
            .m_max0 = m * 2,
            .allocator = allocator,
            .stats = .{
                .lock_contentions = std.atomic.Value(u64).init(0),
                .distance_computations = std.atomic.Value(u64).init(0),
            },
        };
    }

    pub fn deinit(self: *ParallelGraph) void {
        for (self.neighbors) |node_neighbors| {
            for (node_neighbors) |*list| {
                list.deinit(self.allocator);
            }
            self.allocator.free(node_neighbors);
        }
        self.allocator.free(self.neighbors);
        self.allocator.free(self.node_states);
        self.* = undefined;
    }

    /// Try to set entry point if not set or if new node has higher layer.
    pub fn trySetEntryPoint(self: *ParallelGraph, node_id: u32, layer: i32) void {
        while (true) {
            const current = self.entry_point.load(.acquire);
            if (current == INVALID_ENTRY) {
                if (self.entry_point.cmpxchgWeak(current, node_id, .acq_rel, .acquire)) |_| {
                    continue;
                }
                _ = self.max_layer.fetchMax(layer, .acq_rel);
                return;
            }

            const current_layer = self.node_states[current].target_layer;
            if (layer > current_layer) {
                if (self.entry_point.cmpxchgWeak(current, node_id, .acq_rel, .acquire)) |_| {
                    continue;
                }
                _ = self.max_layer.fetchMax(layer, .acq_rel);
            }
            return;
        }
    }

    /// Add a neighbor to a node at a specific layer (thread-safe).
    pub fn addNeighbor(self: *ParallelGraph, node: u32, neighbor: u32, layer: usize) !void {
        const state = &self.node_states[node];

        if (!state.mutex.tryLock()) {
            _ = self.stats.lock_contentions.fetchAdd(1, .monotonic);
            state.mutex.lock();
        }
        defer state.mutex.unlock();

        if (layer < self.neighbors[node].len) {
            try self.neighbors[node][layer].append(self.allocator, neighbor);
        }
    }

    /// Get neighbors of a node at a specific layer (may return stale data during construction).
    pub fn getNeighbors(self: *ParallelGraph, node: u32, layer: usize) []const u32 {
        if (layer >= self.neighbors[node].len) {
            return &[_]u32{};
        }
        return self.neighbors[node][layer].items;
    }

    /// Set neighbors for a node at a layer (replaces existing, thread-safe).
    pub fn setNeighbors(self: *ParallelGraph, node: u32, layer: usize, new_neighbors: []const u32) !void {
        const state = &self.node_states[node];

        if (!state.mutex.tryLock()) {
            _ = self.stats.lock_contentions.fetchAdd(1, .monotonic);
            state.mutex.lock();
        }
        defer state.mutex.unlock();

        if (layer < self.neighbors[node].len) {
            self.neighbors[node][layer].clearRetainingCapacity();
            try self.neighbors[node][layer].appendSlice(self.allocator, new_neighbors);
        }
    }

    /// Convert parallel graph to standard HnswIndex.
    pub fn toHnswIndex(self: *ParallelGraph, allocator: std.mem.Allocator) !hnsw.HnswIndex {
        const node_count = self.node_states.len;

        const nodes = try allocator.alloc(hnsw.HnswIndex.NodeLayers, node_count);
        errdefer allocator.free(nodes);

        for (nodes, self.neighbors) |*node, parallel_neighbors| {
            const layer_count = parallel_neighbors.len;
            node.layers = try allocator.alloc(index_mod.NeighborList, layer_count);

            for (node.layers, parallel_neighbors) |*list, parallel_list| {
                list.nodes = try allocator.dupe(u32, parallel_list.items);
            }
        }

        const entry_pt = self.entry_point.load(.acquire);

        return .{
            .m = self.m,
            .m_max = self.m_max,
            .m_max0 = self.m_max0,
            .ef_construction = 200,
            .entry_point = if (entry_pt == INVALID_ENTRY) null else entry_pt,
            .max_layer = self.max_layer.load(.acquire),
            .nodes = nodes,
            .state_pool = null,
            .distance_cache = null,
            .gpu_accelerator = null,
            .allocator = allocator,
        };
    }
};

// ============================================================================
// Worker Thread
// ============================================================================

/// Per-worker state for parallel insert.
pub const WorkerState = struct {
    worker_id: usize,
    graph: *ParallelGraph,
    records: []const index_mod.VectorRecordView,
    ef_construction: usize,
    allocator: std.mem.Allocator,
    /// Local statistics
    nodes_inserted: usize = 0,
    distances_computed: usize = 0,
    work_stolen: usize = 0,

    /// Insert a single node into the graph.
    pub fn insertNode(self: *WorkerState, node_id: u32) !void {
        const state = &self.graph.node_states[node_id];
        const target_layer = state.target_layer;

        // Mark as inserted (atomic)
        if (state.inserted.swap(true, .acq_rel)) {
            return; // Already inserted by another worker
        }

        // Update entry point if needed
        self.graph.trySetEntryPoint(node_id, target_layer);

        // Get current entry point
        const entry = self.graph.entry_point.load(.acquire);
        if (entry == ParallelGraph.INVALID_ENTRY or entry == node_id) {
            self.nodes_inserted += 1;
            return;
        }

        var curr_node = entry;
        var curr_dist = self.computeDistance(node_id, curr_node);

        // 1. Greedy search down to target layer
        var lc = self.graph.max_layer.load(.acquire);
        while (lc > target_layer) : (lc -= 1) {
            var changed = true;
            while (changed) {
                changed = false;
                const neighbors = self.graph.getNeighbors(curr_node, @intCast(lc));

                for (neighbors) |neighbor| {
                    const d = self.computeDistance(node_id, neighbor);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_node = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 2. Insert at each layer from target_layer down to 0
        lc = @min(target_layer, self.graph.max_layer.load(.acquire));
        while (lc >= 0) : (lc -= 1) {
            try self.connectNeighbors(node_id, curr_node, @intCast(lc));
        }

        self.nodes_inserted += 1;
    }

    /// Connect node to neighbors at a layer.
    fn connectNeighbors(self: *WorkerState, node_id: u32, entry: u32, layer: usize) !void {
        const m_val = if (layer == 0) self.graph.m_max0 else self.graph.m_max;

        // Build candidate list using BFS expansion
        var candidates = std.AutoHashMap(u32, f32).init(self.allocator);
        defer candidates.deinit();

        var visited = std.AutoHashMap(u32, void).init(self.allocator);
        defer visited.deinit();

        var queue = std.ArrayListUnmanaged(u32).empty;
        defer queue.deinit(self.allocator);

        // Start with entry point
        const entry_dist = self.computeDistance(node_id, entry);
        try candidates.put(entry, entry_dist);
        try visited.put(entry, {});
        try queue.append(self.allocator, entry);

        var head: usize = 0;
        while (head < queue.items.len and candidates.count() < self.ef_construction) : (head += 1) {
            const curr = queue.items[head];
            const neighbors = self.graph.getNeighbors(curr, layer);

            for (neighbors) |neighbor| {
                if (!visited.contains(neighbor)) {
                    try visited.put(neighbor, {});
                    const dist = self.computeDistance(node_id, neighbor);
                    try candidates.put(neighbor, dist);
                    try queue.append(self.allocator, neighbor);
                }
            }
        }

        // Select best neighbors
        const selected = try self.selectNeighborsHeuristic(node_id, &candidates, m_val);
        defer self.allocator.free(selected);

        // Set neighbors for this node
        try self.graph.setNeighbors(node_id, layer, selected);

        // Update bidirectional links
        for (selected) |neighbor| {
            const neighbor_links = self.graph.getNeighbors(neighbor, layer);

            var needs_add = true;
            for (neighbor_links) |existing| {
                if (existing == node_id) {
                    needs_add = false;
                    break;
                }
            }

            if (needs_add) {
                if (neighbor_links.len >= m_val) {
                    // Need to prune
                    var link_candidates = std.AutoHashMap(u32, f32).init(self.allocator);
                    defer link_candidates.deinit();

                    for (neighbor_links) |existing| {
                        const dist = self.computeDistance(neighbor, existing);
                        try link_candidates.put(existing, dist);
                    }
                    const new_dist = self.computeDistance(neighbor, node_id);
                    try link_candidates.put(node_id, new_dist);

                    const pruned = try self.selectNeighborsHeuristic(neighbor, &link_candidates, m_val);
                    defer self.allocator.free(pruned);

                    try self.graph.setNeighbors(neighbor, layer, pruned);
                } else {
                    try self.graph.addNeighbor(neighbor, node_id, layer);
                }
            }
        }
    }

    /// Select neighbors using heuristic pruning.
    fn selectNeighborsHeuristic(
        self: *WorkerState,
        node_id: u32,
        candidates: *std.AutoHashMap(u32, f32),
        m_val: usize,
    ) ![]u32 {
        const Pair = struct { id: u32, dist: f32 };
        var sorted = std.ArrayListUnmanaged(Pair).empty;
        defer sorted.deinit(self.allocator);

        var it = candidates.iterator();
        while (it.next()) |e| {
            if (e.key_ptr.* != node_id) {
                try sorted.append(self.allocator, .{ .id = e.key_ptr.*, .dist = e.value_ptr.* });
            }
        }

        std.mem.sort(Pair, sorted.items, {}, struct {
            fn lessThan(_: void, a: Pair, b: Pair) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        // Select with diversity heuristic
        var selected_list = std.ArrayListUnmanaged(u32).empty;
        errdefer selected_list.deinit(self.allocator);

        for (sorted.items) |candidate| {
            if (selected_list.items.len >= m_val) break;

            var should_add = true;
            for (selected_list.items) |existing| {
                const dist_to_existing = 1.0 - simd.cosineSimilarity(
                    self.records[candidate.id].vector,
                    self.records[existing].vector,
                );
                if (dist_to_existing < candidate.dist) {
                    should_add = false;
                    break;
                }
            }

            if (should_add) {
                try selected_list.append(self.allocator, candidate.id);
            }
        }

        // Fill with closest if needed
        if (selected_list.items.len < m_val) {
            for (sorted.items) |candidate| {
                if (selected_list.items.len >= m_val) break;

                var already_added = false;
                for (selected_list.items) |existing| {
                    if (existing == candidate.id) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added) {
                    try selected_list.append(self.allocator, candidate.id);
                }
            }
        }

        return selected_list.toOwnedSlice(self.allocator);
    }

    /// Compute distance between two nodes.
    fn computeDistance(self: *WorkerState, a: u32, b: u32) f32 {
        self.distances_computed += 1;
        return 1.0 - simd.cosineSimilarity(
            self.records[a].vector,
            self.records[b].vector,
        );
    }
};
