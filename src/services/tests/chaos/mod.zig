//! Chaos Testing Framework for ABI
//!
//! Provides chaos engineering tools for production-grade reliability testing.
//! Tests system behavior under various failure conditions including:
//! - Memory allocation failures
//! - Network partitions
//! - Disk write failures
//! - Latency injection
//! - CPU pressure simulation
//! - Random crashes
//!
//! ## Usage
//!
//! ```zig
//! const chaos = @import("chaos/mod.zig");
//!
//! var ctx = try chaos.ChaosContext.init(allocator, 12345);
//! defer ctx.deinit();
//!
//! // Configure fault injection
//! try ctx.addFault(.{
//!     .fault_type = .memory_allocation_failure,
//!     .probability = 0.05,  // 5% failure rate
//! });
//! ctx.enable();
//! defer ctx.disable();
//!
//! // Get a failing allocator that randomly fails
//! var failing_alloc = ctx.getFailingAllocator(allocator);
//!
//! // Use failing_alloc.allocator() in your tests
//! ```
//!
//! ## Reproducibility
//!
//! All chaos tests use seeded random number generators for reproducibility.
//! The same seed will produce the same sequence of failures.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi");
const time = abi.shared.time;

/// Types of faults that can be injected
pub const FaultType = enum {
    /// Randomly fail memory allocations
    memory_allocation_failure,
    /// Simulate network partition (block messages)
    network_partition,
    /// Simulate disk write failures
    disk_write_failure,
    /// Inject latency into operations
    latency_injection,
    /// Simulate CPU pressure (busy-wait delays)
    cpu_pressure,
    /// Randomly cause panic (for recovery testing)
    random_crash,
    /// Corrupt data during transmission
    data_corruption,
    /// Simulate timeout conditions
    timeout_injection,
    /// Drop messages randomly
    message_drop,
    /// Reorder messages (for consensus testing)
    message_reorder,
};

/// Configuration for a specific fault injection rule
pub const FaultConfig = struct {
    /// Type of fault to inject
    fault_type: FaultType,
    /// Probability of fault occurring (0.0 to 1.0)
    probability: f32 = 0.1,
    /// Duration of fault effect in milliseconds (for latency/pressure)
    duration_ms: u64 = 1000,
    /// Target module (null = all modules)
    target_module: ?[]const u8 = null,
    /// Minimum number of successful operations before faults start
    warmup_ops: u32 = 0,
    /// Maximum number of faults to inject (0 = unlimited)
    max_faults: u32 = 0,
    /// Burst mode: inject multiple faults in sequence
    burst_count: u32 = 1,
    /// Recovery time between bursts in milliseconds
    burst_recovery_ms: u64 = 0,
};

/// Statistics about injected faults
pub const FaultStats = struct {
    /// Total number of fault checks performed
    total_checks: u64 = 0,
    /// Total number of faults actually injected
    faults_injected: u64 = 0,
    /// Faults by type
    faults_by_type: [@typeInfo(FaultType).@"enum".fields.len]u64 = [_]u64{0} ** @typeInfo(FaultType).@"enum".fields.len,
    /// Time of first fault injection (nanoseconds since epoch)
    first_fault_time_ns: ?i128 = null,
    /// Time of last fault injection
    last_fault_time_ns: ?i128 = null,
    /// Longest time between faults
    max_fault_interval_ns: u64 = 0,

    pub fn format(
        self: FaultStats,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("ChaosStats{{ checks={d}, faults={d}, rate={d:.2}% }}", .{
            self.total_checks,
            self.faults_injected,
            if (self.total_checks > 0)
                @as(f64, @floatFromInt(self.faults_injected)) / @as(f64, @floatFromInt(self.total_checks)) * 100.0
            else
                0.0,
        });
    }
};

/// Event emitted when a fault is injected (for logging/debugging)
pub const FaultEvent = struct {
    fault_type: FaultType,
    timestamp_ns: i128,
    module: ?[]const u8,
    details: []const u8,
};

/// Callback type for fault events
pub const FaultEventCallback = *const fn (FaultEvent) void;

/// Chaos testing context that manages fault injection
pub const ChaosContext = struct {
    allocator: std.mem.Allocator,
    faults: std.ArrayListUnmanaged(FaultConfig),
    active: bool,
    rng: std.Random.DefaultPrng,
    stats: FaultStats,
    mutex: sync.Mutex,
    operation_count: u64,
    event_callback: ?FaultEventCallback,
    /// Injected faults per type (for max_faults tracking)
    faults_per_config: std.ArrayListUnmanaged(u64),

    const Self = @This();

    /// Initialize a new chaos context with a seed for reproducibility
    pub fn init(allocator: std.mem.Allocator, seed: u64) Self {
        return .{
            .allocator = allocator,
            .faults = .{},
            .active = false,
            .rng = std.Random.DefaultPrng.init(seed),
            .stats = .{},
            .mutex = .{},
            .operation_count = 0,
            .event_callback = null,
            .faults_per_config = .{},
        };
    }

    /// Deinitialize the chaos context
    pub fn deinit(self: *Self) void {
        self.faults.deinit(self.allocator);
        self.faults_per_config.deinit(self.allocator);
        self.* = undefined;
    }

    /// Enable chaos mode - faults will start being injected
    pub fn enable(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.active = true;
        self.operation_count = 0;
    }

    /// Disable chaos mode - no faults will be injected
    pub fn disable(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.active = false;
    }

    /// Check if chaos mode is active
    pub fn isActive(self: *Self) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.active;
    }

    /// Add a fault injection rule
    pub fn addFault(self: *Self, config: FaultConfig) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.faults.append(self.allocator, config);
        try self.faults_per_config.append(self.allocator, 0);
    }

    /// Remove all fault rules
    pub fn clearFaults(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.faults.clearRetainingCapacity();
        self.faults_per_config.clearRetainingCapacity();
    }

    /// Set a callback for fault events (useful for debugging/logging)
    pub fn setEventCallback(self: *Self, callback: ?FaultEventCallback) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.event_callback = callback;
    }

    /// Check if a fault should be triggered for the given type
    pub fn shouldFault(self: *Self, fault_type: FaultType) bool {
        return self.shouldFaultWithModule(fault_type, null);
    }

    /// Check if a fault should be triggered for the given type and module
    pub fn shouldFaultWithModule(self: *Self, fault_type: FaultType, module: ?[]const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.active) return false;

        self.operation_count += 1;
        self.stats.total_checks += 1;

        for (self.faults.items, 0..) |config, i| {
            if (config.fault_type != fault_type) continue;

            // Check module filter
            if (config.target_module) |target| {
                if (module) |m| {
                    if (!std.mem.eql(u8, target, m)) continue;
                } else {
                    continue;
                }
            }

            // Check warmup period
            if (self.operation_count <= config.warmup_ops) continue;

            // Check max faults limit
            if (config.max_faults > 0 and self.faults_per_config.items[i] >= config.max_faults) continue;

            // Roll for probability
            const roll = self.rng.random().float(f32);
            if (roll < config.probability) {
                self.recordFault(fault_type, module, i);
                return true;
            }
        }

        return false;
    }

    fn recordFault(self: *Self, fault_type: FaultType, module: ?[]const u8, config_idx: usize) void {
        const now = blk: {
            var timer = time.Timer.start() catch break :blk 0;
            break :blk @as(i128, timer.read());
        };

        self.stats.faults_injected += 1;
        self.stats.faults_by_type[@intFromEnum(fault_type)] += 1;
        self.faults_per_config.items[config_idx] += 1;

        if (self.stats.first_fault_time_ns == null) {
            self.stats.first_fault_time_ns = now;
        }

        if (self.stats.last_fault_time_ns) |last| {
            const interval: u64 = @intCast(@max(0, now - last));
            if (interval > self.stats.max_fault_interval_ns) {
                self.stats.max_fault_interval_ns = interval;
            }
        }
        self.stats.last_fault_time_ns = now;

        // Emit event if callback is set
        if (self.event_callback) |callback| {
            callback(.{
                .fault_type = fault_type,
                .timestamp_ns = now,
                .module = module,
                .details = "Fault injected",
            });
        }
    }

    /// Get current fault statistics
    pub fn getStats(self: *Self) FaultStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Reset statistics
    pub fn resetStats(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.stats = .{};
        for (self.faults_per_config.items) |*count| {
            count.* = 0;
        }
    }

    /// Get a failing allocator that randomly fails allocations based on chaos config
    pub fn getFailingAllocator(self: *Self, backing: std.mem.Allocator) FailingAllocator {
        return FailingAllocator.init(self, backing);
    }

    /// Inject latency if configured (call at potential latency injection points)
    pub fn maybeInjectLatency(self: *Self) void {
        if (self.shouldFault(.latency_injection)) {
            self.injectLatencyInternal();
        }
    }

    fn injectLatencyInternal(self: *Self) void {
        // Find the latency config
        for (self.faults.items) |config| {
            if (config.fault_type == .latency_injection) {
                self.sleepMs(config.duration_ms);
                return;
            }
        }
    }

    /// Inject CPU pressure if configured
    pub fn maybeInjectCpuPressure(self: *Self) void {
        if (self.shouldFault(.cpu_pressure)) {
            self.injectCpuPressureInternal();
        }
    }

    fn injectCpuPressureInternal(self: *Self) void {
        // Find the CPU pressure config
        for (self.faults.items) |config| {
            if (config.fault_type == .cpu_pressure) {
                // Busy-wait to simulate CPU pressure
                const end_time = std.time.milliTimestamp() + @as(i64, @intCast(config.duration_ms));
                while (std.time.milliTimestamp() < end_time) {
                    // Busy spin
                    var i: u32 = 0;
                    while (i < 1000) : (i += 1) {
                        std.atomic.spinLoopHint();
                    }
                }
                return;
            }
        }
    }

    /// Check if data should be corrupted (returns true if corruption should occur)
    pub fn shouldCorruptData(self: *Self) bool {
        return self.shouldFault(.data_corruption);
    }

    /// Check if a message should be dropped
    pub fn shouldDropMessage(self: *Self) bool {
        return self.shouldFault(.message_drop);
    }

    /// Check if a timeout should be injected
    pub fn shouldTimeout(self: *Self) bool {
        return self.shouldFault(.timeout_injection);
    }

    /// Check if a disk write should fail
    pub fn shouldFailDiskWrite(self: *Self) bool {
        return self.shouldFault(.disk_write_failure);
    }

    /// Check if a network partition should occur
    pub fn shouldNetworkPartition(self: *Self) bool {
        return self.shouldFault(.network_partition);
    }

    // Platform-aware sleep - delegates to shared time module
    fn sleepMs(self: *Self, ms: u64) void {
        _ = self;
        time.sleepMs(ms);
    }
};

/// Allocator wrapper that randomly fails allocations based on chaos context
pub const FailingAllocator = struct {
    backing: std.mem.Allocator,
    chaos: *ChaosContext,

    const Self = @This();

    pub fn init(chaos: *ChaosContext, backing: std.mem.Allocator) Self {
        return .{
            .backing = backing,
            .chaos = chaos,
        };
    }

    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Check if we should fail this allocation
        if (self.chaos.shouldFault(.memory_allocation_failure)) {
            return null;
        }

        return self.backing.rawAlloc(len, ptr_align, ret_addr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Check if we should fail this resize
        if (new_len > buf.len and self.chaos.shouldFault(.memory_allocation_failure)) {
            return false;
        }

        return self.backing.rawResize(buf, buf_align, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Check if we should fail this remap (expansion)
        if (new_len > memory.len and self.chaos.shouldFault(.memory_allocation_failure)) {
            return null;
        }

        return self.backing.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        // Never fail frees
        self.backing.rawFree(buf, buf_align, ret_addr);
    }
};

/// Network partition simulator for distributed system testing
pub const NetworkPartitionSimulator = struct {
    chaos: *ChaosContext,
    /// Partitioned node pairs (node_a, node_b) - communication blocked between them
    partitions: std.ArrayListUnmanaged([2][]const u8),
    allocator: std.mem.Allocator,
    mutex: sync.Mutex,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, chaos: *ChaosContext) Self {
        return .{
            .chaos = chaos,
            .partitions = .{},
            .allocator = allocator,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.partitions.items) |pair| {
            self.allocator.free(pair[0]);
            self.allocator.free(pair[1]);
        }
        self.partitions.deinit(self.allocator);
        self.* = undefined;
    }

    /// Create a partition between two nodes
    pub fn partition(self: *Self, node_a: []const u8, node_b: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const a_copy = try self.allocator.dupe(u8, node_a);
        errdefer self.allocator.free(a_copy);
        const b_copy = try self.allocator.dupe(u8, node_b);
        errdefer self.allocator.free(b_copy);

        try self.partitions.append(self.allocator, [2][]const u8{ a_copy, b_copy });
    }

    /// Heal a partition between two nodes
    pub fn heal(self: *Self, node_a: []const u8, node_b: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var i: usize = 0;
        while (i < self.partitions.items.len) {
            const pair = self.partitions.items[i];
            const matches = (std.mem.eql(u8, pair[0], node_a) and std.mem.eql(u8, pair[1], node_b)) or
                (std.mem.eql(u8, pair[0], node_b) and std.mem.eql(u8, pair[1], node_a));
            if (matches) {
                self.allocator.free(pair[0]);
                self.allocator.free(pair[1]);
                _ = self.partitions.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Heal all partitions
    pub fn healAll(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.partitions.items) |pair| {
            self.allocator.free(pair[0]);
            self.allocator.free(pair[1]);
        }
        self.partitions.clearRetainingCapacity();
    }

    /// Check if communication is blocked between two nodes
    pub fn isPartitioned(self: *Self, node_a: []const u8, node_b: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Also check chaos-based random partition
        if (self.chaos.shouldNetworkPartition()) {
            return true;
        }

        for (self.partitions.items) |pair| {
            const matches = (std.mem.eql(u8, pair[0], node_a) and std.mem.eql(u8, pair[1], node_b)) or
                (std.mem.eql(u8, pair[0], node_b) and std.mem.eql(u8, pair[1], node_a));
            if (matches) return true;
        }
        return false;
    }
};

/// Message delay/reorder simulator for testing message ordering issues
pub const MessageDelaySimulator = struct {
    chaos: *ChaosContext,
    /// Delayed messages with their delivery time
    delayed: std.ArrayListUnmanaged(DelayedMessage),
    allocator: std.mem.Allocator,
    mutex: sync.Mutex,
    next_id: u64,

    pub const DelayedMessage = struct {
        id: u64,
        deliver_at_ms: i64,
        payload: []const u8,
        from: []const u8,
        to: []const u8,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, chaos: *ChaosContext) Self {
        return .{
            .chaos = chaos,
            .delayed = .{},
            .allocator = allocator,
            .mutex = .{},
            .next_id = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.delayed.items) |msg| {
            self.allocator.free(msg.payload);
            self.allocator.free(msg.from);
            self.allocator.free(msg.to);
        }
        self.delayed.deinit(self.allocator);
        self.* = undefined;
    }

    /// Potentially delay a message. Returns null if message should be delivered now.
    /// Returns message ID if delayed.
    pub fn maybeDelay(self: *Self, from: []const u8, to: []const u8, payload: []const u8) !?u64 {
        if (!self.chaos.shouldFault(.message_reorder)) {
            return null;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        const delay_ms: u64 = self.chaos.rng.random().intRangeAtMost(u64, 10, 500);
        const deliver_at = std.time.milliTimestamp() + @as(i64, @intCast(delay_ms));

        const id = self.next_id;
        self.next_id += 1;

        const payload_copy = try self.allocator.dupe(u8, payload);
        errdefer self.allocator.free(payload_copy);
        const from_copy = try self.allocator.dupe(u8, from);
        errdefer self.allocator.free(from_copy);
        const to_copy = try self.allocator.dupe(u8, to);
        errdefer self.allocator.free(to_copy);

        try self.delayed.append(self.allocator, .{
            .id = id,
            .deliver_at_ms = deliver_at,
            .payload = payload_copy,
            .from = from_copy,
            .to = to_copy,
        });

        return id;
    }

    /// Get messages ready for delivery
    pub fn getReadyMessages(self: *Self) ![]DelayedMessage {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.milliTimestamp();
        var ready = std.ArrayListUnmanaged(DelayedMessage){};

        var i: usize = 0;
        while (i < self.delayed.items.len) {
            if (self.delayed.items[i].deliver_at_ms <= now) {
                try ready.append(self.allocator, self.delayed.swapRemove(i));
            } else {
                i += 1;
            }
        }

        return try ready.toOwnedSlice(self.allocator);
    }

    /// Get count of pending delayed messages
    pub fn pendingCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.delayed.items.len;
    }
};

/// Test helper to run a function with chaos enabled and verify recovery
pub fn runWithChaos(
    allocator: std.mem.Allocator,
    seed: u64,
    faults: []const FaultConfig,
    test_fn: *const fn (*ChaosContext) anyerror!void,
) !ChaosTestResult {
    var chaos = ChaosContext.init(allocator, seed);
    defer chaos.deinit();

    for (faults) |fault| {
        try chaos.addFault(fault);
    }

    chaos.enable();

    const start_time = blk: {
        var timer = time.Timer.start() catch break :blk 0;
        break :blk @as(i128, timer.read());
    };
    var test_error: ?anyerror = null;

    test_fn(&chaos) catch |err| {
        test_error = err;
    };

    chaos.disable();
    const end_time = blk: {
        var timer = time.Timer.start() catch break :blk 0;
        break :blk @as(i128, timer.read());
    };

    return .{
        .success = test_error == null,
        .error_value = test_error,
        .stats = chaos.getStats(),
        .duration_ns = @intCast(@max(0, end_time - start_time)),
        .seed = seed,
    };
}

/// Result of running a chaos test
pub const ChaosTestResult = struct {
    success: bool,
    error_value: ?anyerror,
    stats: FaultStats,
    duration_ns: u64,
    seed: u64,

    pub fn format(
        self: ChaosTestResult,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const status = if (self.success) "PASS" else "FAIL";
        try writer.print("ChaosTest[{s}]: seed={d}, duration={d}ms, {}", .{
            status,
            self.seed,
            self.duration_ns / std.time.ns_per_ms,
            self.stats,
        });
    }
};

// ============================================================================
// Sub-module exports
// ============================================================================

pub const ha_chaos_test = @import("ha_chaos_test.zig");
pub const database_chaos_test = @import("database_chaos_test.zig");
pub const network_chaos_test = @import("network_chaos_test.zig");

// ============================================================================
// Tests
// ============================================================================

test "ChaosContext initialization" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 12345);
    defer chaos.deinit();

    try std.testing.expect(!chaos.isActive());
    chaos.enable();
    try std.testing.expect(chaos.isActive());
    chaos.disable();
    try std.testing.expect(!chaos.isActive());
}

test "ChaosContext fault injection" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 12345);
    defer chaos.deinit();

    try chaos.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 1.0, // Always fail
    });
    chaos.enable();

    // Should always fault with 100% probability
    try std.testing.expect(chaos.shouldFault(.memory_allocation_failure));

    const stats = chaos.getStats();
    try std.testing.expect(stats.faults_injected >= 1);
}

test "ChaosContext probability-based faults" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 42);
    defer chaos.deinit();

    try chaos.addFault(.{
        .fault_type = .latency_injection,
        .probability = 0.5, // 50% chance
    });
    chaos.enable();

    var fault_count: u32 = 0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        if (chaos.shouldFault(.latency_injection)) {
            fault_count += 1;
        }
    }

    // With 50% probability over 100 tries, expect roughly 30-70 faults
    // (allowing for random variation)
    try std.testing.expect(fault_count > 20);
    try std.testing.expect(fault_count < 80);
}

test "ChaosContext warmup period" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 12345);
    defer chaos.deinit();

    try chaos.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 1.0,
        .warmup_ops = 5,
    });
    chaos.enable();

    // First 5 operations should not fault
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        try std.testing.expect(!chaos.shouldFault(.memory_allocation_failure));
    }

    // After warmup, should fault
    try std.testing.expect(chaos.shouldFault(.memory_allocation_failure));
}

test "ChaosContext max faults limit" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 12345);
    defer chaos.deinit();

    try chaos.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 1.0,
        .max_faults = 3,
    });
    chaos.enable();

    // Should fault exactly 3 times
    try std.testing.expect(chaos.shouldFault(.memory_allocation_failure));
    try std.testing.expect(chaos.shouldFault(.memory_allocation_failure));
    try std.testing.expect(chaos.shouldFault(.memory_allocation_failure));

    // Fourth and beyond should not fault
    try std.testing.expect(!chaos.shouldFault(.memory_allocation_failure));
    try std.testing.expect(!chaos.shouldFault(.memory_allocation_failure));

    const stats = chaos.getStats();
    try std.testing.expectEqual(@as(u64, 3), stats.faults_injected);
}

test "FailingAllocator integration" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 12345);
    defer chaos.deinit();

    try chaos.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 0.0, // Never fail initially
    });
    chaos.enable();

    var failing = chaos.getFailingAllocator(allocator);
    const fa = failing.allocator();

    // Should succeed when probability is 0
    const buf = try fa.alloc(u8, 100);
    fa.free(buf);

    // Now set to always fail
    chaos.clearFaults();
    try chaos.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 1.0,
    });

    // Should fail
    const result = fa.alloc(u8, 100);
    try std.testing.expect(result == error.OutOfMemory);
}

test "NetworkPartitionSimulator" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 12345);
    defer chaos.deinit();

    var sim = NetworkPartitionSimulator.init(allocator, &chaos);
    defer sim.deinit();

    // No partition initially
    try std.testing.expect(!sim.isPartitioned("node-1", "node-2"));

    // Create partition
    try sim.partition("node-1", "node-2");
    try std.testing.expect(sim.isPartitioned("node-1", "node-2"));
    try std.testing.expect(sim.isPartitioned("node-2", "node-1")); // Bidirectional

    // Other pairs should not be partitioned
    try std.testing.expect(!sim.isPartitioned("node-1", "node-3"));

    // Heal partition
    sim.heal("node-1", "node-2");
    try std.testing.expect(!sim.isPartitioned("node-1", "node-2"));
}

test "FaultStats tracking" {
    const allocator = std.testing.allocator;
    var chaos = ChaosContext.init(allocator, 12345);
    defer chaos.deinit();

    try chaos.addFault(.{
        .fault_type = .memory_allocation_failure,
        .probability = 1.0,
    });
    try chaos.addFault(.{
        .fault_type = .latency_injection,
        .probability = 1.0,
    });
    chaos.enable();

    _ = chaos.shouldFault(.memory_allocation_failure);
    _ = chaos.shouldFault(.memory_allocation_failure);
    _ = chaos.shouldFault(.latency_injection);

    const stats = chaos.getStats();
    try std.testing.expectEqual(@as(u64, 3), stats.total_checks);
    try std.testing.expectEqual(@as(u64, 3), stats.faults_injected);
    try std.testing.expectEqual(@as(u64, 2), stats.faults_by_type[@intFromEnum(FaultType.memory_allocation_failure)]);
    try std.testing.expectEqual(@as(u64, 1), stats.faults_by_type[@intFromEnum(FaultType.latency_injection)]);
}
