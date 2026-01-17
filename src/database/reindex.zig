//! Automatic re-indexing for vector databases with background monitoring.
const std = @import("std");
const time = @import("../shared/utils/time.zig");
const index = @import("index.zig");

pub const ReindexConfig = struct {
    enabled: bool = true,
    fragmentation_threshold: f64 = 0.30,
    check_interval_ms: u64 = 30000,
    min_records_for_check: usize = 100,
    max_reindex_threads: usize = 4,
    batch_size: usize = 10000,
};

pub const ReindexState = enum {
    idle,
    checking,
    reindexing,
    paused,
};

pub const ReindexMetrics = struct {
    total_checks: u64 = 0,
    total_reindexes: u64 = 0,
    fragmented_triggers: u64 = 0,
    last_check_time: i64 = 0,
    last_reindex_duration_ns: u64 = 0,
    current_state: ReindexState = .idle,
};

pub const AutoReindexer = struct {
    allocator: std.mem.Allocator,
    config: ReindexConfig,
    metrics: ReindexMetrics,
    state: ReindexState,
    running: std.atomic.Value(bool),
    thread: ?std.Thread = null,
    condition: std.Thread.Condition = .{},
    mutex: std.Thread.Mutex = .{},
    index_manager: *index.IndexManager,
    wake_event: std.Thread.Wake = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        config: ReindexConfig,
        index_manager: *index.IndexManager,
    ) !AutoReindexer {
        return .{
            .allocator = allocator,
            .config = config,
            .metrics = .{},
            .state = .idle,
            .running = std.atomic.Value(bool).init(false),
            .thread = null,
            .condition = .{},
            .mutex = .{},
            .index_manager = index_manager,
            .wake_event = .{},
        };
    }

    pub fn deinit(self: *AutoReindexer) void {
        self.stop();
        self.wake_event.deinit();
        self.* = undefined;
    }

    pub fn start(self: *AutoReindexer) !void {
        if (self.running.load(.acquire)) return;
        self.running.store(true, .release);
        self.thread = try std.Thread.spawn(.{}, runMonitor, .{self});
    }

    pub fn stop(self: *AutoReindexer) void {
        if (!self.running.load(.acquire)) return;
        self.running.store(false, .release);
        self.wake_event.signal();
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    pub fn triggerImmediateCheck(self: *AutoReindexer) void {
        self.wake_event.signal();
    }

    fn runMonitor(self: *AutoReindexer) void {
        while (self.running.load(.acquire)) {
            {
                const lock = self.mutex.acquire();
                defer lock.release();
                if (!self.running.load(.acquire)) break;
            }

            if (self.config.enabled) {
                self.checkAndReindex();
            }

            const sleep_duration = std.time.ns_per_ms * self.config.check_interval_ms;
            _ = self.wake_event.timedWait(sleep_duration);
        }
    }

    fn checkAndReindex(self: *AutoReindexer) void {
        self.mutex.acquire();
        defer self.mutex.release();

        self.metrics.total_checks += 1;
        self.metrics.last_check_time = time.unixSeconds();
        self.metrics.current_state = .checking;

        const fragmentation = self.calculateFragmentation();
        if (fragmentation < self.config.fragmentation_threshold) {
            self.metrics.current_state = .idle;
            return;
        }

        self.metrics.fragmented_triggers += 1;
        self.performReindex();
    }

    fn calculateFragmentation(self: *AutoReindexer) f64 {
        const total = self.index_manager.record_count;
        if (total == 0) return 0.0;
        if (self.index_manager.index == null) return 1.0;

        return self.estimateDeletedRatio();
    }

    /// Estimate the ratio of deleted/invalid records in the index.
    /// Uses index structure analysis to determine fragmentation.
    fn estimateDeletedRatio(self: *AutoReindexer) f64 {
        const idx_manager = self.index_manager;
        const total_records = idx_manager.record_count;

        if (total_records == 0) return 0.0;
        if (idx_manager.index == null) return 0.0;

        // Analyze index structure for fragmentation markers
        var empty_slots: usize = 0;
        var total_slots: usize = 0;

        const idx = idx_manager.index.?;
        switch (idx.*) {
            .hnsw => |hnsw| {
                // Count empty neighbor slots as potential deleted nodes
                total_slots = hnsw.neighbors.len;
                for (hnsw.neighbors) |neighbor_list| {
                    if (neighbor_list.nodes.len == 0) {
                        empty_slots += 1;
                    }
                }
            },
            .ivf_pq => |ivf| {
                // Count empty clusters as fragmentation
                total_slots = ivf.clusters.len;
                for (ivf.clusters) |cluster| {
                    if (cluster.members.len == 0) {
                        empty_slots += 1;
                    }
                }
            },
        }

        if (total_slots == 0) return 0.0;

        // Calculate ratio of empty slots to total
        const ratio = @as(f64, @floatFromInt(empty_slots)) / @as(f64, @floatFromInt(total_slots));

        // Apply heuristic: actual deleted ratio is likely higher than empty slot ratio
        return @min(ratio * 1.5, 1.0);
    }

    fn performReindex(self: *AutoReindexer) void {
        self.metrics.current_state = .reindexing;
        var timer = std.time.Timer.start() catch {
            self.metrics.current_state = .idle;
            return;
        };

        const allocator = self.allocator;
        const config = self.config;
        const idx_manager = self.index_manager;

        const records = self.collectAllRecords() catch {
            self.metrics.current_state = .idle;
            return;
        };
        defer allocator.free(records);

        if (records.len < config.min_records_for_check) {
            self.metrics.current_state = .idle;
            return;
        }

        idx_manager.setConfig(idx_manager.config);
        idx_manager.markDirty();

        idx_manager.buildIfNeeded(allocator, records) catch {
            return;
        };

        self.metrics.last_reindex_duration_ns = timer.read();
        self.metrics.total_reindexes += 1;
        self.metrics.current_state = .idle;
    }

    /// Collect all valid records from the index for rebuilding.
    /// This extracts active records by scanning the index structure.
    fn collectAllRecords(self: *AutoReindexer) ![]index.VectorRecordView {
        const idx_manager = self.index_manager;
        const allocator = self.allocator;

        // If no index exists, return empty
        const idx = idx_manager.index orelse return &.{};

        var records = std.ArrayListUnmanaged(index.VectorRecordView){};
        errdefer records.deinit(allocator);

        switch (idx.*) {
            .hnsw => |hnsw| {
                // Collect records from HNSW index nodes
                for (hnsw.neighbors, 0..) |neighbor_list, node_id| {
                    // Skip empty nodes (likely deleted)
                    if (neighbor_list.nodes.len == 0) continue;

                    // Get the vector for this node
                    if (node_id < hnsw.vectors.len) {
                        const vector = hnsw.vectors[node_id];
                        if (vector.len > 0) {
                            try records.append(allocator, .{
                                .id = @intCast(node_id),
                                .vector = vector,
                            });
                        }
                    }
                }
            },
            .ivf_pq => |ivf| {
                // Collect records from IVF-PQ clusters
                for (ivf.clusters) |cluster| {
                    for (cluster.members) |member_id| {
                        // Look up vector by member ID
                        if (member_id < ivf.codes.len) {
                            // Reconstruct approximate vector from codes
                            // For now, add a placeholder
                            // Real implementation would decode PQ codes
                            try records.append(allocator, .{
                                .id = @intCast(member_id),
                                .vector = &.{}, // Placeholder - would decode from codes
                            });
                        }
                    }
                }
            },
        }

        return try records.toOwnedSlice(allocator);
    }

    pub fn getMetrics(self: *const AutoReindexer) ReindexMetrics {
        return self.metrics;
    }

    pub fn getState(self: *const AutoReindexer) ReindexState {
        return self.state;
    }
};

pub const FragmentationStats = struct {
    total_records: usize,
    deleted_records: usize,
    fragmentation_ratio: f64,
    index_size_bytes: usize,
    estimated_search_overhead: f64,
};

pub fn calculateFragmentationStats(
    index_manager: *const index.IndexManager,
    deleted_count: usize,
) FragmentationStats {
    const total = index_manager.record_count;
    const deleted = deleted_count;
    const ratio = if (total > 0) @as(f64, @floatFromInt(deleted)) / @as(f64, @floatFromInt(total)) else 0.0;

    var index_size: usize = 0;
    if (index_manager.index) |idx| {
        switch (idx.*) {
            .hnsw => |hnsw| {
                index_size = hnsw.neighbors.len * @sizeOf(index.NeighborList);
                for (hnsw.neighbors) |list| {
                    index_size += list.nodes.len * @sizeOf(u32);
                }
            },
            .ivf_pq => |ivf| {
                index_size = ivf.codes.len;
                for (ivf.clusters) |cluster| {
                    index_size += cluster.centroid.len * @sizeOf(f32);
                    index_size += cluster.members.len * @sizeOf(u32);
                }
            },
        }
    }

    const search_overhead = if (ratio > 0.3) ratio * 2.0 else ratio;

    return .{
        .total_records = total,
        .deleted_records = deleted,
        .fragmentation_ratio = ratio,
        .index_size_bytes = index_size,
        .estimated_search_overhead = search_overhead,
    };
}

pub const ReindexScheduler = struct {
    allocator: std.mem.Allocator,
    reindexers: std.ArrayListUnmanaged(*AutoReindexer),
    global_config: ReindexConfig,

    pub fn init(allocator: std.mem.Allocator, config: ReindexConfig) ReindexScheduler {
        return .{
            .allocator = allocator,
            .reindexers = std.ArrayListUnmanaged(*AutoReindexer).empty,
            .global_config = config,
        };
    }

    pub fn deinit(self: *ReindexScheduler) void {
        for (self.reindexers.items) |reindexer| {
            reindexer.deinit();
            self.allocator.destroy(reindexer);
        }
        self.reindexers.deinit(self.allocator);
    }

    pub fn addIndex(self: *ReindexScheduler, index_manager: *index.IndexManager) !*AutoReindexer {
        const reindexer = try self.allocator.create(AutoReindexer);
        errdefer self.allocator.destroy(reindexer);

        reindexer.* = try AutoReindexer.init(self.allocator, self.global_config, index_manager);
        errdefer reindexer.deinit();

        try self.reindexers.append(self.allocator, reindexer);
        return reindexer;
    }

    pub fn startAll(self: *ReindexScheduler) !void {
        for (self.reindexers.items) |reindexer| {
            try reindexer.start();
        }
    }

    pub fn stopAll(self: *ReindexScheduler) void {
        for (self.reindexers.items) |reindexer| {
            reindexer.stop();
        }
    }

    pub fn triggerAll(self: *ReindexScheduler) void {
        for (self.reindexers.items) |reindexer| {
            reindexer.triggerImmediateCheck();
        }
    }

    pub fn getTotalMetrics(self: *const ReindexScheduler) ReindexMetrics {
        var total = ReindexMetrics{};
        for (self.reindexers.items) |reindexer| {
            const metrics = reindexer.getMetrics();
            total.total_checks += metrics.total_checks;
            total.total_reindexes += metrics.total_reindexes;
            total.fragmented_triggers += metrics.fragmented_triggers;
        }
        return total;
    }
};

test "auto reindexer initialization" {
    const allocator = std.testing.allocator;
    var index_manager = index.IndexManager.init(.{ .index_type = .hnsw });
    defer index_manager.deinit(allocator);

    var reindexer = try AutoReindexer.init(allocator, .{}, &index_manager);
    defer reindexer.deinit();

    try std.testing.expect(!reindexer.running.load(.acquire));
    try std.testing.expectEqual(ReindexState.idle, reindexer.state);
}

test "reindex scheduler add and remove" {
    const allocator = std.testing.allocator;
    var scheduler = ReindexScheduler.init(allocator, .{});
    defer scheduler.deinit();

    var index_manager = index.IndexManager.init(.{ .index_type = .ivf_pq });
    defer index_manager.deinit(allocator);

    const reindexer = try scheduler.addIndex(&index_manager);
    try std.testing.expectEqual(@as(usize, 1), scheduler.reindexers.items.len);

    try std.testing.expectEqual(ReindexState.idle, reindexer.getState());
}

test "fragmentation stats calculation" {
    var index_manager = index.IndexManager.init(.{ .index_type = .hnsw });
    defer index_manager.deinit(std.testing.allocator);

    const stats = calculateFragmentationStats(&index_manager, 10);
    try std.testing.expectEqual(@as(usize, 0), stats.total_records);
    try std.testing.expectEqual(@as(usize, 10), stats.deleted_records);
    try std.testing.expectEqual(true, stats.fragmentation_ratio > 0);
}
