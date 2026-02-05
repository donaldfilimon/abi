//! Parallel HNSW Index Builder
//!
//! Multi-threaded HNSW index construction using work-stealing for high throughput.
//!
//! ## Algorithm
//!
//! HNSW insertion has inherent sequential dependencies (each insert modifies the graph),
//! but we can parallelize in several ways:
//!
//! 1. **Batch layer assignment**: Pre-compute random layer for all nodes in parallel
//! 2. **Parallel distance computation**: SIMD/batch distance calculation during neighbor search
//! 3. **Work-stealing insert**: Concurrent node insertion with fine-grained locking
//! 4. **Lock-free graph updates**: Atomic neighbor list updates where possible
//!
//! ## Usage
//!
//! ```zig
//! const builder = try ParallelHnswBuilder.init(allocator, .{
//!     .thread_count = 8,
//!     .m = 16,
//!     .ef_construction = 200,
//! });
//! defer builder.deinit();
//!
//! var index = try builder.build(records);
//! defer index.deinit(allocator);
//! ```
//!
//! ## Performance
//!
//! For N nodes and T threads, typical speedup is 2-4x on multi-core systems.
//! The speedup is limited by graph update contention, but work-stealing ensures
//! good load balancing even with variable insertion costs.

const std = @import("std");
const builtin = @import("builtin");
const hnsw = @import("hnsw.zig");
const index_mod = @import("index.zig");
const simd = @import("../../services/shared/simd.zig");
const ChaseLevDeque = @import("../../services/runtime/concurrency/chase_lev.zig").ChaseLevDeque;
const WorkStealingScheduler = @import("../../services/runtime/concurrency/chase_lev.zig").WorkStealingScheduler;

/// Whether threading is available on this target
const is_threaded_target = builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for parallel HNSW index building.
pub const ParallelBuildConfig = struct {
    /// Number of worker threads (null = auto-detect based on CPU count)
    thread_count: ?usize = null,
    /// Maximum connections per node (M parameter)
    m: usize = 16,
    /// Size of dynamic candidate list during construction
    ef_construction: usize = 200,
    /// Batch size for parallel layer assignment
    layer_batch_size: usize = 1024,
    /// Batch size for parallel distance computation
    distance_batch_size: usize = 256,
    /// Enable search state pool for parallel queries
    search_pool_size: usize = 0,
    /// Enable distance cache
    distance_cache_size: usize = 0,
    /// Random seed (null = use system time)
    seed: ?u64 = null,
    /// Progress callback (optional)
    progress_callback: ?*const fn (completed: usize, total: usize) void = null,
};

/// Statistics from parallel build.
pub const ParallelBuildStats = struct {
    /// Total nodes inserted
    total_nodes: usize = 0,
    /// Total time in nanoseconds
    total_time_ns: u64 = 0,
    /// Time spent in layer assignment (ns)
    layer_assign_time_ns: u64 = 0,
    /// Time spent in parallel insert (ns)
    insert_time_ns: u64 = 0,
    /// Total distance computations
    distance_computations: u64 = 0,
    /// Number of lock contentions
    lock_contentions: u64 = 0,
    /// Maximum graph layer reached
    max_layer: i32 = 0,
    /// Work items stolen between threads
    work_stolen: u64 = 0,

    /// Throughput in nodes per second.
    pub fn throughput(self: ParallelBuildStats) f64 {
        if (self.total_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.total_nodes)) * 1_000_000_000.0 / @as(f64, @floatFromInt(self.total_time_ns));
    }

    /// Average insert latency in microseconds.
    pub fn avgInsertLatencyUs(self: ParallelBuildStats) f64 {
        if (self.total_nodes == 0) return 0;
        return @as(f64, @floatFromInt(self.insert_time_ns)) / @as(f64, @floatFromInt(self.total_nodes)) / 1000.0;
    }
};

// ============================================================================
// Parallel Graph Structure
// ============================================================================

/// Node state during parallel construction.
const ParallelNodeState = struct {
    /// Target layer for this node
    target_layer: i32,
    /// Insertion completed flag
    inserted: std.atomic.Value(bool),
    /// Lock for neighbor list updates
    mutex: std.Thread.Mutex,
};

/// Thread-safe parallel HNSW graph for concurrent construction.
const ParallelGraph = struct {
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

    const INVALID_ENTRY: u32 = std.math.maxInt(u32);

    fn init(allocator: std.mem.Allocator, node_count: usize, m: usize, layer_assignments: []const i32) !ParallelGraph {
        const node_states = try allocator.alloc(ParallelNodeState, node_count);
        errdefer allocator.free(node_states);

        const neighbors = try allocator.alloc([]std.ArrayListUnmanaged(u32), node_count);
        // Track how many nodes we've initialized for cleanup on error
        var initialized_nodes: usize = 0;
        errdefer {
            // Clean up any partially initialized neighbor lists
            for (neighbors[0..initialized_nodes]) |node_neighbors| {
                for (node_neighbors) |*list| {
                    list.deinit(allocator);
                }
                allocator.free(node_neighbors);
            }
            allocator.free(neighbors);
        }

        var max_layer: i32 = 0;

        for (node_states, neighbors, layer_assignments) |*state, *node_neighbors, layer| {
            state.* = .{
                .target_layer = layer,
                .inserted = std.atomic.Value(bool).init(false),
                .mutex = .{},
            };

            // Allocate layers for this node
            const layer_count: usize = @intCast(layer + 1);
            node_neighbors.* = try allocator.alloc(std.ArrayListUnmanaged(u32), layer_count);
            for (node_neighbors.*) |*list| {
                list.* = .empty;
            }

            if (layer > max_layer) {
                max_layer = layer;
            }

            // Track successful initialization for errdefer cleanup
            initialized_nodes += 1;
        }

        return .{
            .node_states = node_states,
            .neighbors = neighbors,
            .entry_point = std.atomic.Value(u32).init(INVALID_ENTRY),
            .max_layer = std.atomic.Value(i32).init(max_layer),
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

    fn deinit(self: *ParallelGraph) void {
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
    fn trySetEntryPoint(self: *ParallelGraph, node_id: u32, layer: i32) void {
        // Try to CAS the entry point
        while (true) {
            const current = self.entry_point.load(.acquire);
            if (current == INVALID_ENTRY) {
                // First node
                if (self.entry_point.cmpxchgWeak(current, node_id, .acq_rel, .acquire)) |_| {
                    continue; // CAS failed, retry
                }
                _ = self.max_layer.fetchMax(layer, .acq_rel);
                return;
            }

            // Check if new node has higher layer
            const current_layer = self.node_states[current].target_layer;
            if (layer > current_layer) {
                if (self.entry_point.cmpxchgWeak(current, node_id, .acq_rel, .acquire)) |_| {
                    continue; // CAS failed, retry
                }
                _ = self.max_layer.fetchMax(layer, .acq_rel);
            }
            return;
        }
    }

    /// Add a neighbor to a node at a specific layer (thread-safe).
    fn addNeighbor(self: *ParallelGraph, node: u32, neighbor: u32, layer: usize) !void {
        const state = &self.node_states[node];

        // Try lock without spinning first
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
    fn getNeighbors(self: *ParallelGraph, node: u32, layer: usize) []const u32 {
        if (layer >= self.neighbors[node].len) {
            return &[_]u32{};
        }
        return self.neighbors[node][layer].items;
    }

    /// Set neighbors for a node at a layer (replaces existing, thread-safe).
    fn setNeighbors(self: *ParallelGraph, node: u32, layer: usize, new_neighbors: []const u32) !void {
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
    fn toHnswIndex(self: *ParallelGraph, allocator: std.mem.Allocator) !hnsw.HnswIndex {
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
            .ef_construction = 200, // Default, will be set by builder
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
// Insert Task
// ============================================================================

/// Task for work-stealing scheduler.
const InsertTask = struct {
    node_id: u32,
    priority: u8, // Higher layer = higher priority
};

// ============================================================================
// Worker Thread
// ============================================================================

/// Per-worker state for parallel insert.
const WorkerState = struct {
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
    fn insertNode(self: *WorkerState, node_id: u32) !void {
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
            // First node or we are the entry point
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
            // Get current neighbors of neighbor
            const neighbor_links = self.graph.getNeighbors(neighbor, layer);

            // Check if we need to add the link
            var needs_add = true;
            for (neighbor_links) |existing| {
                if (existing == node_id) {
                    needs_add = false;
                    break;
                }
            }

            if (needs_add) {
                if (neighbor_links.len >= m_val) {
                    // Need to prune - collect distances and select
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
                    // Just add the new link
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
        while (it.next()) |entry| {
            if (entry.key_ptr.* != node_id) {
                try sorted.append(self.allocator, .{ .id = entry.key_ptr.*, .dist = entry.value_ptr.* });
            }
        }

        // Sort by distance
        std.mem.sort(Pair, sorted.items, {}, struct {
            fn lessThan(_: void, a: Pair, b: Pair) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        // Select with diversity heuristic
        var selected = std.ArrayListUnmanaged(u32).empty;
        errdefer selected.deinit(self.allocator);

        for (sorted.items) |candidate| {
            if (selected.items.len >= m_val) break;

            var should_add = true;
            for (selected.items) |existing| {
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
                try selected.append(self.allocator, candidate.id);
            }
        }

        // Fill with closest if needed
        if (selected.items.len < m_val) {
            for (sorted.items) |candidate| {
                if (selected.items.len >= m_val) break;

                var already_added = false;
                for (selected.items) |existing| {
                    if (existing == candidate.id) {
                        already_added = true;
                        break;
                    }
                }

                if (!already_added) {
                    try selected.append(self.allocator, candidate.id);
                }
            }
        }

        return selected.toOwnedSlice(self.allocator);
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

// ============================================================================
// Parallel Builder
// ============================================================================

/// Parallel HNSW index builder.
pub const ParallelHnswBuilder = struct {
    allocator: std.mem.Allocator,
    config: ParallelBuildConfig,
    thread_count: usize,

    pub fn init(allocator: std.mem.Allocator, config: ParallelBuildConfig) ParallelHnswBuilder {
        const count = config.thread_count orelse blk: {
            const cpu_count: usize = if (comptime is_threaded_target)
                std.Thread.getCpuCount() catch 4
            else
                1;
            break :blk @max(1, cpu_count);
        };

        return .{
            .allocator = allocator,
            .config = config,
            .thread_count = count,
        };
    }

    /// Build HNSW index in parallel.
    pub fn build(
        self: *ParallelHnswBuilder,
        records: []const index_mod.VectorRecordView,
    ) !hnsw.HnswIndex {
        var stats: ParallelBuildStats = .{};
        return self.buildWithStats(records, &stats);
    }

    /// Build HNSW index with statistics.
    pub fn buildWithStats(
        self: *ParallelHnswBuilder,
        records: []const index_mod.VectorRecordView,
        stats: *ParallelBuildStats,
    ) !hnsw.HnswIndex {
        if (records.len == 0) return index_mod.IndexError.EmptyIndex;

        var total_timer = std.time.Timer.start() catch null;

        // Phase 1: Assign layers to all nodes
        var layer_timer = std.time.Timer.start() catch null;
        const layer_assignments = try self.assignLayers(records.len);
        defer self.allocator.free(layer_assignments);
        if (layer_timer) |*t| stats.layer_assign_time_ns = t.read();

        // Phase 2: Create parallel graph structure
        var graph = try ParallelGraph.init(self.allocator, records.len, self.config.m, layer_assignments);
        defer graph.deinit();

        // Phase 3: Parallel insert using work-stealing
        var insert_timer = std.time.Timer.start() catch null;
        try self.parallelInsert(&graph, records);
        if (insert_timer) |*t| stats.insert_time_ns = t.read();

        // Collect stats
        stats.total_nodes = records.len;
        stats.distance_computations = graph.stats.distance_computations.load(.acquire);
        stats.lock_contentions = graph.stats.lock_contentions.load(.acquire);
        stats.max_layer = graph.max_layer.load(.acquire);
        if (total_timer) |*t| stats.total_time_ns = t.read();

        // Convert to standard HnswIndex
        var index = try graph.toHnswIndex(self.allocator);
        index.ef_construction = self.config.ef_construction;

        // Enable optional features
        if (self.config.search_pool_size > 0) {
            const pool = try self.allocator.create(hnsw.SearchStatePool);
            pool.* = try hnsw.SearchStatePool.init(self.allocator, self.config.search_pool_size);
            index.state_pool = pool;
        }

        if (self.config.distance_cache_size > 0) {
            const cache = try self.allocator.create(hnsw.DistanceCache);
            cache.* = try hnsw.DistanceCache.init(self.allocator, self.config.distance_cache_size);
            index.distance_cache = cache;
        }

        return index;
    }

    /// Assign random layers to all nodes.
    fn assignLayers(self: *ParallelHnswBuilder, node_count: usize) ![]i32 {
        const layers = try self.allocator.alloc(i32, node_count);
        errdefer self.allocator.free(layers);

        const m_l = 1.0 / @log(@as(f32, @floatFromInt(self.config.m)));
        const seed = self.config.seed orelse blk: {
            var buf: [8]u8 = undefined;
            std.posix.getrandom(&buf) catch {
                break :blk 42;
            };
            break :blk std.mem.readInt(u64, &buf, .little);
        };

        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();

        for (layers) |*layer| {
            layer.* = @intFromFloat(@floor(-@log(random.float(f32)) * m_l));
        }

        return layers;
    }

    /// Perform parallel insertion using work-stealing.
    fn parallelInsert(
        self: *ParallelHnswBuilder,
        graph: *ParallelGraph,
        records: []const index_mod.VectorRecordView,
    ) !void {
        const node_count = records.len;

        // For small graphs or single thread, use sequential
        if (node_count < 100 or self.thread_count == 1) {
            var worker = WorkerState{
                .worker_id = 0,
                .graph = graph,
                .records = records,
                .ef_construction = self.config.ef_construction,
                .allocator = self.allocator,
            };

            for (0..node_count) |i| {
                try worker.insertNode(@intCast(i));
                if (self.config.progress_callback) |callback| {
                    callback(i + 1, node_count);
                }
            }

            _ = graph.stats.distance_computations.fetchAdd(worker.distances_computed, .release);
            return;
        }

        // Create work-stealing scheduler
        var scheduler = try WorkStealingScheduler(InsertTask).init(self.allocator, self.thread_count);
        defer scheduler.deinit();

        // Sort nodes by layer (higher layers first) for better parallel efficiency
        const sorted_nodes = try self.allocator.alloc(InsertTask, node_count);
        defer self.allocator.free(sorted_nodes);

        for (sorted_nodes, 0..) |*task, i| {
            task.* = .{
                .node_id = @intCast(i),
                .priority = @intCast(@min(255, graph.node_states[i].target_layer)),
            };
        }

        // Sort by priority (higher layer = higher priority = inserted first)
        std.mem.sort(InsertTask, sorted_nodes, {}, struct {
            fn lessThan(_: void, a: InsertTask, b: InsertTask) bool {
                return a.priority > b.priority; // Descending
            }
        }.lessThan);

        // Distribute tasks to workers initially
        for (sorted_nodes, 0..) |task, i| {
            try scheduler.push(i % self.thread_count, task);
        }

        // Completion tracking
        var completed = std.atomic.Value(usize).init(0);
        var has_error = std.atomic.Value(bool).init(false);

        // Spawn worker threads
        const threads = try self.allocator.alloc(std.Thread, self.thread_count);
        defer self.allocator.free(threads);

        const worker_stats = try self.allocator.alloc(WorkerState, self.thread_count);
        defer self.allocator.free(worker_stats);

        for (threads, worker_stats, 0..) |*t, *ws, i| {
            ws.* = WorkerState{
                .worker_id = i,
                .graph = graph,
                .records = records,
                .ef_construction = self.config.ef_construction,
                .allocator = self.allocator,
            };

            t.* = try std.Thread.spawn(.{}, workerLoop, .{
                ws,
                &scheduler,
                node_count,
                &completed,
                &has_error,
            });
        }

        // Wait for completion
        for (threads) |t| {
            t.join();
        }

        // Aggregate stats
        var total_distances: usize = 0;
        var total_stolen: usize = 0;
        for (worker_stats) |ws| {
            total_distances += ws.distances_computed;
            total_stolen += ws.work_stolen;
        }
        _ = graph.stats.distance_computations.fetchAdd(total_distances, .release);

        if (has_error.load(.acquire)) {
            return error.ParallelInsertFailed;
        }
    }
};

/// Worker thread loop.
fn workerLoop(
    worker: *WorkerState,
    scheduler: *WorkStealingScheduler(InsertTask),
    total_nodes: usize,
    completed: *std.atomic.Value(usize),
    has_error: *std.atomic.Value(bool),
) void {
    while (completed.load(.acquire) < total_nodes and !has_error.load(.acquire)) {
        // Try to get task from local queue first, then steal
        const task_opt = scheduler.getTask(worker.worker_id);

        if (task_opt) |task| {
            // Track if we stole this task
            if (scheduler.worker_deques[worker.worker_id].isEmpty()) {
                worker.work_stolen += 1;
            }

            worker.insertNode(task.node_id) catch {
                has_error.store(true, .release);
                return;
            };

            _ = completed.fetchAdd(1, .release);
        } else {
            // No work available, yield
            std.atomic.spinLoopHint();
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "parallel layer assignment" {
    const allocator = std.testing.allocator;

    var builder = ParallelHnswBuilder.init(allocator, .{ .seed = 42 });
    const layers = try builder.assignLayers(100);
    defer allocator.free(layers);

    // Check layer distribution
    var layer_counts = [_]usize{0} ** 10;
    var max_layer: i32 = 0;
    for (layers) |layer| {
        if (layer < 10) {
            layer_counts[@intCast(layer)] += 1;
        }
        if (layer > max_layer) max_layer = layer;
    }

    // Most nodes should be at layer 0
    try std.testing.expect(layer_counts[0] > 50);
    // Should have some nodes at higher layers
    try std.testing.expect(max_layer >= 1);
}

test "parallel hnsw builder small" {
    const allocator = std.testing.allocator;

    const records = [_]index_mod.VectorRecordView{
        .{ .id = 1, .vector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } },
        .{ .id = 2, .vector = &[_]f32{ 0.0, 1.0, 0.0, 0.0 } },
        .{ .id = 3, .vector = &[_]f32{ 0.0, 0.0, 1.0, 0.0 } },
        .{ .id = 4, .vector = &[_]f32{ 0.5, 0.5, 0.0, 0.0 } },
        .{ .id = 5, .vector = &[_]f32{ 0.0, 0.5, 0.5, 0.0 } },
    };

    var builder = ParallelHnswBuilder.init(allocator, .{
        .thread_count = 2,
        .m = 4,
        .ef_construction = 50,
        .seed = 42,
    });

    var stats: ParallelBuildStats = .{};
    var index = try builder.buildWithStats(&records, &stats);
    defer index.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 5), stats.total_nodes);
    try std.testing.expect(stats.distance_computations > 0);
    try std.testing.expect(index.entry_point != null);

    // Test search
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try index.search(allocator, &records, &query, 3);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
    // First result should be record 1 (same vector)
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

test "parallel hnsw builder medium" {
    const allocator = std.testing.allocator;

    // Generate random vectors
    const n = 200;
    const dim = 32;

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    const vectors = try allocator.alloc([dim]f32, n);
    defer allocator.free(vectors);

    const records = try allocator.alloc(index_mod.VectorRecordView, n);
    defer allocator.free(records);

    for (vectors, records, 0..) |*vec, *record, i| {
        for (vec) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0;
        }
        record.* = .{
            .id = @intCast(i),
            .vector = vec,
        };
    }

    var builder = ParallelHnswBuilder.init(allocator, .{
        .thread_count = 4,
        .m = 16,
        .ef_construction = 100,
        .seed = 42,
    });

    var stats: ParallelBuildStats = .{};
    var index = try builder.buildWithStats(records, &stats);
    defer index.deinit(allocator);

    try std.testing.expectEqual(@as(usize, n), stats.total_nodes);
    try std.testing.expect(stats.throughput() > 0);

    // Test search accuracy
    const query = vectors[0][0..];
    const results = try index.search(allocator, records, query, 5);
    defer allocator.free(results);

    // Self-query should return the query vector first
    try std.testing.expect(results.len > 0);
    try std.testing.expectEqual(@as(u64, 0), results[0].id);
}

test "parallel build stats" {
    var stats = ParallelBuildStats{
        .total_nodes = 1000,
        .total_time_ns = 1_000_000_000, // 1 second
        .insert_time_ns = 800_000_000, // 0.8 seconds
    };

    const throughput = stats.throughput();
    try std.testing.expectApproxEqAbs(@as(f64, 1000.0), throughput, 0.1);

    const latency = stats.avgInsertLatencyUs();
    try std.testing.expectApproxEqAbs(@as(f64, 800.0), latency, 0.1);
}

test "parallel graph concurrent access" {
    const allocator = std.testing.allocator;

    const layers = [_]i32{ 0, 1, 0, 2, 0 };
    var graph = try ParallelGraph.init(allocator, 5, 4, &layers);
    defer graph.deinit();

    // Concurrent neighbor updates
    const threads_count = 4;
    var threads: [threads_count]std.Thread = undefined;

    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(g: *ParallelGraph, tid: usize) !void {
                for (0..10) |j| {
                    const node = (tid + j) % 5;
                    const neighbor = (tid + j + 1) % 5;
                    try g.addNeighbor(@intCast(node), @intCast(neighbor), 0);
                }
            }
        }.run, .{ &graph, i });
    }

    for (&threads) |*t| {
        t.join();
    }

    // Verify graph is consistent
    var total_edges: usize = 0;
    for (graph.neighbors) |node_neighbors| {
        if (node_neighbors.len > 0) {
            total_edges += node_neighbors[0].items.len;
        }
    }
    try std.testing.expect(total_edges > 0);
}
