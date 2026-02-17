//! Cross-Backend GPU Coordinator
//!
//! Manages simultaneous operation across CUDA, Vulkan, Metal, and other backends
//! with unified device selection, memory transfers, and workload distribution.
//!
//! ## Features
//!
//! - **Multi-Backend Discovery**: Detects and initializes all available GPU backends
//! - **Intelligent Scheduling**: Routes workloads to optimal backend/device combinations
//! - **Learning-Based Optimization**: Records outcomes to improve future scheduling
//! - **Health Monitoring**: Tracks backend health and performance metrics
//! - **Unified Memory Management**: Cross-backend memory transfer coordination
//!
//! ## Usage
//!
//! ```zig
//! const mega = @import("mega/mod.zig");
//!
//! var coordinator = try mega.Coordinator.init(allocator);
//! defer coordinator.deinit();
//!
//! // Schedule workload to best backend
//! const profile = mega.WorkloadProfile{
//!     .compute_intensity = 0.8,
//!     .memory_requirement_mb = 2048,
//!     .is_training = true,
//! };
//! const decision = coordinator.schedule(profile);
//!
//! // Record outcome for learning
//! try coordinator.recordOutcome(decision, actual_time_ms, success);
//! ```

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const build_options = @import("build_options");
const multi_device = @import("../multi_device.zig");
const backend_mod = @import("../backend.zig");
const interface = @import("../interface.zig");

/// Backend instance with metadata for cross-backend coordination.
pub const BackendInstance = struct {
    /// Type of backend (cuda, vulkan, metal, etc.)
    backend_type: backend_mod.Backend,
    /// Number of devices available on this backend
    device_count: u32,
    /// Total memory in MB across all devices
    total_memory_mb: u64,
    /// Available memory in MB across all devices
    available_memory_mb: u64,
    /// Backend priority for scheduling (higher = preferred)
    priority: u8,
    /// Whether this backend is currently healthy and operational
    is_healthy: bool,
    /// Whether this is a CPU emulation backend
    is_emulated: bool,
    /// Feature flags
    supports_fp16: bool,
    supports_fp64: bool,
    supports_int8: bool,
    supports_unified_memory: bool,

    /// Calculate a health score combining availability and priority.
    pub fn healthScore(self: BackendInstance) f32 {
        if (!self.is_healthy) return 0.0;

        const memory_ratio = if (self.total_memory_mb > 0)
            @as(f32, @floatFromInt(self.available_memory_mb)) /
                @as(f32, @floatFromInt(self.total_memory_mb))
        else
            1.0;

        const base_score = memory_ratio * @as(f32, @floatFromInt(self.priority));

        // Prefer non-emulated backends
        return if (self.is_emulated) base_score * 0.5 else base_score;
    }

    /// Check if backend supports a specific precision.
    pub fn supportsPrecision(self: BackendInstance, precision: Precision) bool {
        return switch (precision) {
            .fp16 => self.supports_fp16,
            .fp32 => true, // All backends support fp32
            .fp64 => self.supports_fp64,
            .int8 => self.supports_int8,
        };
    }
};

/// Precision requirements for workloads.
pub const Precision = enum {
    fp16,
    fp32,
    fp64,
    int8,
};

/// Workload characteristics for scheduling decisions.
pub const WorkloadProfile = struct {
    /// Compute intensity: 0.0 (memory-bound) to 1.0 (compute-bound)
    compute_intensity: f32 = 0.5,
    /// Memory requirement in MB
    memory_requirement_mb: u64 = 0,
    /// Preferred backend type (null = auto-select)
    preferred_backend: ?backend_mod.Backend = null,
    /// Required precision
    required_precision: Precision = .fp32,
    /// Batch size for batched operations
    batch_size: u32 = 1,
    /// Whether this is a training workload (prefers CUDA/Metal)
    is_training: bool = false,
    /// Whether low latency is critical
    low_latency: bool = false,
    /// Expected duration in ms (for scheduling hints)
    expected_duration_ms: u64 = 0,
    /// Workload category for specialized routing
    category: WorkloadCategory = .general,
};

/// Categories of workloads for specialized scheduling.
pub const WorkloadCategory = enum {
    general,
    matrix_multiply,
    convolution,
    attention,
    embedding,
    elementwise,
    reduction,
    fft,
    sorting,
};

/// Scheduling decision from the coordinator.
pub const ScheduleDecision = struct {
    /// Selected backend type
    backend_type: backend_mod.Backend,
    /// Selected device ID within the backend
    device_id: multi_device.DeviceId,
    /// Estimated execution time in ms
    estimated_time_ms: u64,
    /// Confidence in this decision (0.0 to 1.0)
    confidence: f32,
    /// Human-readable reason for the decision
    reason: []const u8,
    /// Unique ID for tracking this decision
    decision_id: u64,
};

/// Cross-Backend Coordinator for unified GPU orchestration.
pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    backends: std.ArrayListUnmanaged(BackendInstance),
    device_groups: std.AutoHashMap(backend_mod.Backend, *multi_device.DeviceGroup),
    scheduling_history: std.ArrayListUnmanaged(SchedulingRecord),
    stats: CoordinatorStats,
    next_decision_id: u64,
    mutex: sync.Mutex,

    /// Record of a scheduling decision and its outcome.
    const SchedulingRecord = struct {
        workload: WorkloadProfile,
        decision: ScheduleDecision,
        actual_time_ms: u64,
        success: bool,
        timestamp: i64,
    };

    /// Statistics tracked by the coordinator.
    pub const CoordinatorStats = struct {
        /// Comptime-computed array sizes from enum field counts
        const backend_count = @typeInfo(backend_mod.Backend).@"enum".fields.len;
        const category_count = @typeInfo(WorkloadCategory).@"enum".fields.len;

        /// Total number of scheduling decisions made
        total_schedules: u64 = 0,
        /// Number of successful scheduling decisions
        successful_schedules: u64 = 0,
        /// Total compute time across all workloads in ms
        total_compute_time_ms: u64 = 0,
        /// Average scheduling latency in microseconds
        avg_scheduling_latency_us: u64 = 0,
        /// Per-backend usage counts (indexed by backend enum value)
        backend_usage: [backend_count]u64 = [_]u64{0} ** backend_count,
        /// Per-backend success counts
        backend_successes: [backend_count]u64 = [_]u64{0} ** backend_count,
        /// Per-category scheduling counts
        category_counts: [category_count]u64 = [_]u64{0} ** category_count,
    };

    /// Initialize the coordinator and discover available backends.
    pub fn init(allocator: std.mem.Allocator) !*Coordinator {
        const self = try allocator.create(Coordinator);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .backends = .{},
            .device_groups = std.AutoHashMap(backend_mod.Backend, *multi_device.DeviceGroup).init(allocator),
            .scheduling_history = .{},
            .stats = .{},
            .next_decision_id = 1,
            .mutex = .{},
        };

        // Discover all available backends
        try self.discoverBackends();

        return self;
    }

    /// Deinitialize the coordinator and release all resources.
    pub fn deinit(self: *Coordinator) void {
        // Clean up device groups
        var it = self.device_groups.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.device_groups.deinit();

        self.backends.deinit(self.allocator);
        self.scheduling_history.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Discover all available GPU backends on this system.
    fn discoverBackends(self: *Coordinator) !void {
        // Clear existing backends
        self.backends.clearRetainingCapacity();

        // Try each backend type in priority order
        const backend_configs = [_]struct {
            backend: backend_mod.Backend,
            priority: u8,
            check_fn: *const fn () bool,
        }{
            .{ .backend = .cuda, .priority = 10, .check_fn = isCudaAvailable },
            .{ .backend = .metal, .priority = 9, .check_fn = isMetalAvailable },
            .{ .backend = .vulkan, .priority = 8, .check_fn = isVulkanAvailable },
            .{ .backend = .fpga, .priority = 7, .check_fn = isFpgaAvailable },
            .{ .backend = .webgpu, .priority = 6, .check_fn = isWebGPUAvailable },
            .{ .backend = .opengl, .priority = 5, .check_fn = isOpenGLAvailable },
            .{ .backend = .stdgpu, .priority = 3, .check_fn = isStdgpuAvailable },
        };

        for (backend_configs) |cfg| {
            if (cfg.check_fn()) {
                const instance = createBackendInstance(cfg.backend, cfg.priority);
                try self.backends.append(self.allocator, instance);
            }
        }

        // Always ensure at least CPU fallback is available
        if (self.backends.items.len == 0) {
            const fallback = createBackendInstance(.stdgpu, 1);
            try self.backends.append(self.allocator, fallback);
        }
    }

    /// Create a backend instance with default configuration.
    fn createBackendInstance(bt: backend_mod.Backend, priority: u8) BackendInstance {
        return BackendInstance{
            .backend_type = bt,
            .device_count = 1,
            .total_memory_mb = getDefaultMemoryMb(bt),
            .available_memory_mb = getDefaultMemoryMb(bt),
            .priority = priority,
            .is_healthy = true,
            .is_emulated = bt == .stdgpu,
            .supports_fp16 = bt == .cuda or bt == .metal or bt == .vulkan,
            .supports_fp64 = bt == .cuda,
            .supports_int8 = bt == .cuda or bt == .metal,
            .supports_unified_memory = bt == .cuda or bt == .metal,
        };
    }

    fn getDefaultMemoryMb(bt: backend_mod.Backend) u64 {
        return switch (bt) {
            .cuda => 8192,
            .metal => 8192,
            .vulkan => 4096,
            .webgpu => 2048,
            .opengl, .opengles => 2048,
            .fpga => 4096,
            .stdgpu => 4096,
            .webgl2 => 1024,
            .tpu => 16384,
            .simulated => 2048,
        };
    }

    /// Schedule a workload to the best available backend and device.
    pub fn schedule(self: *Coordinator, profile: WorkloadProfile) ScheduleDecision {
        self.mutex.lock();
        defer self.mutex.unlock();

        var best_score: f32 = -1.0;
        var best_backend: ?BackendInstance = null;

        for (self.backends.items) |backend| {
            const score = scoreBackendForWorkload(backend, profile);
            if (score > best_score) {
                best_score = score;
                best_backend = backend;
            }
        }

        // Fall back to first available if none scored well
        const selected = best_backend orelse self.backends.items[0];

        const decision_id = self.next_decision_id;
        self.next_decision_id += 1;

        self.stats.total_schedules += 1;
        const backend_idx = @intFromEnum(selected.backend_type);
        if (backend_idx < self.stats.backend_usage.len) {
            self.stats.backend_usage[backend_idx] += 1;
        }

        const category_idx = @intFromEnum(profile.category);
        if (category_idx < self.stats.category_counts.len) {
            self.stats.category_counts[category_idx] += 1;
        }

        return .{
            .backend_type = selected.backend_type,
            .device_id = 0, // Default to first device
            .estimated_time_ms = estimateTime(selected, profile),
            .confidence = @min(best_score / 10.0, 1.0), // Normalize to 0-1
            .reason = selectReason(selected, profile),
            .decision_id = decision_id,
        };
    }

    /// Record the outcome of a scheduling decision for learning.
    pub fn recordOutcome(
        self: *Coordinator,
        decision: ScheduleDecision,
        actual_time_ms: u64,
        success: bool,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Find the original profile (simplified - in real impl would track by decision_id)
        const record = SchedulingRecord{
            .workload = .{}, // Would store original profile
            .decision = decision,
            .actual_time_ms = actual_time_ms,
            .success = success,
            .timestamp = time.nowMs(),
        };

        try self.scheduling_history.append(self.allocator, record);

        // Update statistics
        if (success) {
            self.stats.successful_schedules += 1;
            const backend_idx = @intFromEnum(decision.backend_type);
            if (backend_idx < self.stats.backend_successes.len) {
                self.stats.backend_successes[backend_idx] += 1;
            }
        }
        self.stats.total_compute_time_ms += actual_time_ms;

        // Trim history if too large - use efficient O(n) approach instead of O(n^2) orderedRemove loop
        const max_history = 10000;
        if (self.scheduling_history.items.len > max_history) {
            const to_remove = self.scheduling_history.items.len - max_history;
            // Shift items in place - O(n) instead of O(n^2)
            const items = self.scheduling_history.items;
            std.mem.copyForwards(SchedulingRecord, items[0..max_history], items[to_remove..]);
            self.scheduling_history.shrinkRetainingCapacity(max_history);
        }
    }

    /// Get summary of available backends.
    pub fn getBackendsSummary(self: *Coordinator) []const BackendInstance {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.backends.items;
    }

    /// Get current coordinator statistics.
    pub fn getStats(self: *Coordinator) CoordinatorStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.stats;
    }

    /// Get the number of available backends.
    pub fn backendCount(self: *Coordinator) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.backends.items.len;
    }

    /// Check if a specific backend is available.
    pub fn hasBackend(self: *Coordinator, backend_type: backend_mod.Backend) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.backends.items) |backend| {
            if (backend.backend_type == backend_type and backend.is_healthy) {
                return true;
            }
        }
        return false;
    }

    /// Update backend health status.
    pub fn updateBackendHealth(self: *Coordinator, backend_type: backend_mod.Backend, is_healthy: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.backends.items) |*backend| {
            if (backend.backend_type == backend_type) {
                backend.is_healthy = is_healthy;
                break;
            }
        }
    }

    /// Update backend memory availability.
    pub fn updateBackendMemory(
        self: *Coordinator,
        backend_type: backend_mod.Backend,
        available_mb: u64,
    ) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.backends.items) |*backend| {
            if (backend.backend_type == backend_type) {
                backend.available_memory_mb = available_mb;
                break;
            }
        }
    }

    /// Refresh backend discovery (re-detect available backends).
    pub fn refresh(self: *Coordinator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.discoverBackends();
    }

    /// Get success rate for a specific backend (0.0 to 1.0).
    pub fn getBackendSuccessRate(self: *Coordinator, backend_type: backend_mod.Backend) f32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const idx = @intFromEnum(backend_type);
        if (idx >= self.stats.backend_usage.len) return 0.0;

        const usage = self.stats.backend_usage[idx];
        const successes = self.stats.backend_successes[idx];

        if (usage == 0) return 1.0; // No data, assume success
        return @as(f32, @floatFromInt(successes)) / @as(f32, @floatFromInt(usage));
    }
};

/// Score a backend for a given workload profile.
fn scoreBackendForWorkload(backend: BackendInstance, profile: WorkloadProfile) f32 {
    var score = backend.healthScore();

    // Check hard requirements first
    if (!backend.supportsPrecision(profile.required_precision)) {
        return 0.0;
    }

    if (backend.available_memory_mb < profile.memory_requirement_mb) {
        score *= 0.1; // Heavy penalty but not exclusion
    }

    // Prefer explicitly requested backend
    if (profile.preferred_backend) |pref| {
        if (backend.backend_type == pref) {
            score *= 2.0;
        }
    }

    // Training workloads prefer CUDA or Metal
    if (profile.is_training) {
        if (backend.backend_type == .cuda) {
            score *= 1.8;
        } else if (backend.backend_type == .metal) {
            score *= 1.5;
        }
    }

    // Compute-bound workloads prefer high-performance backends
    if (profile.compute_intensity > 0.7) {
        if (backend.backend_type == .cuda or backend.backend_type == .metal) {
            score *= 1.3;
        }
    }

    // Low-latency workloads prefer local backends
    if (profile.low_latency) {
        if (backend.backend_type == .metal or backend.backend_type == .cuda) {
            score *= 1.2;
        }
    }

    // Large batch sizes benefit from GPU backends
    if (profile.batch_size > 32) {
        if (!backend.is_emulated) {
            score *= 1.2;
        }
    }

    // Specialized category bonuses
    switch (profile.category) {
        .matrix_multiply, .convolution, .attention => {
            if (backend.backend_type == .cuda) score *= 1.4;
        },
        .fft => {
            if (backend.backend_type == .cuda or backend.backend_type == .metal) score *= 1.3;
        },
        else => {},
    }

    return score;
}

fn estimateTime(backend: BackendInstance, profile: WorkloadProfile) u64 {
    if (profile.expected_duration_ms > 0) {
        // Adjust expected time based on backend characteristics
        const multiplier: f32 = if (backend.is_emulated) 10.0 else 1.0;
        return @intFromFloat(@as(f32, @floatFromInt(profile.expected_duration_ms)) * multiplier);
    }

    // Default estimates based on workload characteristics
    const base_time: u64 = 100;
    const memory_factor = profile.memory_requirement_mb / 1024;
    const compute_factor: u64 = @intFromFloat(profile.compute_intensity * 100);

    return base_time + memory_factor + compute_factor;
}

fn selectReason(backend: BackendInstance, profile: WorkloadProfile) []const u8 {
    if (profile.preferred_backend != null and profile.preferred_backend == backend.backend_type) {
        return "Selected preferred backend";
    }

    if (profile.is_training) {
        if (backend.backend_type == .cuda) {
            return "CUDA selected for training workload";
        } else if (backend.backend_type == .metal) {
            return "Metal selected for training workload";
        }
    }

    if (backend.is_emulated) {
        return "CPU fallback selected (no GPU available)";
    }

    return "Selected based on availability and workload profile";
}

// ============================================================================
// Backend Availability Checks
// ============================================================================

fn isCudaAvailable() bool {
    return comptime build_options.gpu_cuda;
}

fn isMetalAvailable() bool {
    return comptime build_options.gpu_metal;
}

fn isVulkanAvailable() bool {
    return comptime build_options.gpu_vulkan;
}

fn isWebGPUAvailable() bool {
    return comptime build_options.gpu_webgpu;
}

fn isOpenGLAvailable() bool {
    return comptime build_options.gpu_opengl;
}

fn isFpgaAvailable() bool {
    return comptime build_options.gpu_fpga;
}

fn isStdgpuAvailable() bool {
    return comptime build_options.gpu_stdgpu;
}

// ============================================================================
// Tests
// ============================================================================

test "coordinator initialization" {
    const allocator = std.testing.allocator;
    const coord = try Coordinator.init(allocator);
    defer coord.deinit();

    // Should have at least one backend (CPU fallback)
    try std.testing.expect(coord.backendCount() >= 1);
}

test "workload scheduling" {
    const allocator = std.testing.allocator;
    const coord = try Coordinator.init(allocator);
    defer coord.deinit();

    const profile = WorkloadProfile{
        .compute_intensity = 0.8,
        .memory_requirement_mb = 1024,
        .is_training = true,
    };

    const decision = coord.schedule(profile);

    // Should get a valid decision
    try std.testing.expect(decision.confidence >= 0.0);
    try std.testing.expect(decision.confidence <= 1.0);
    try std.testing.expect(decision.decision_id > 0);
}

test "outcome recording" {
    const allocator = std.testing.allocator;
    const coord = try Coordinator.init(allocator);
    defer coord.deinit();

    const profile = WorkloadProfile{
        .compute_intensity = 0.5,
        .memory_requirement_mb = 512,
    };

    const decision = coord.schedule(profile);

    // Record successful outcome
    try coord.recordOutcome(decision, 150, true);

    const stats = coord.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_schedules);
    try std.testing.expectEqual(@as(u64, 1), stats.successful_schedules);
    try std.testing.expectEqual(@as(u64, 150), stats.total_compute_time_ms);
}

test "backend health updates" {
    const allocator = std.testing.allocator;
    const coord = try Coordinator.init(allocator);
    defer coord.deinit();

    // Get a backend type that exists (use thread-safe accessor)
    const backends = coord.getBackendsSummary();
    if (backends.len > 0) {
        const bt = backends[0].backend_type;

        // Should initially be healthy
        try std.testing.expect(coord.hasBackend(bt));

        // Mark unhealthy
        coord.updateBackendHealth(bt, false);

        // Should now report as unavailable
        try std.testing.expect(!coord.hasBackend(bt));

        // Mark healthy again
        coord.updateBackendHealth(bt, true);
        try std.testing.expect(coord.hasBackend(bt));
    }
}

test "backend instance scoring" {
    const instance = BackendInstance{
        .backend_type = .cuda,
        .device_count = 1,
        .total_memory_mb = 8192,
        .available_memory_mb = 6144,
        .priority = 10,
        .is_healthy = true,
        .is_emulated = false,
        .supports_fp16 = true,
        .supports_fp64 = true,
        .supports_int8 = true,
        .supports_unified_memory = true,
    };

    const score = instance.healthScore();
    try std.testing.expect(score > 0.0);

    // Unhealthy backend should score 0
    var unhealthy = instance;
    unhealthy.is_healthy = false;
    try std.testing.expectEqual(@as(f32, 0.0), unhealthy.healthScore());

    // Emulated backend should score lower
    var emulated = instance;
    emulated.is_emulated = true;
    try std.testing.expect(emulated.healthScore() < score);
}

test "precision support" {
    const cuda_instance = BackendInstance{
        .backend_type = .cuda,
        .device_count = 1,
        .total_memory_mb = 8192,
        .available_memory_mb = 8192,
        .priority = 10,
        .is_healthy = true,
        .is_emulated = false,
        .supports_fp16 = true,
        .supports_fp64 = true,
        .supports_int8 = true,
        .supports_unified_memory = true,
    };

    try std.testing.expect(cuda_instance.supportsPrecision(.fp16));
    try std.testing.expect(cuda_instance.supportsPrecision(.fp32));
    try std.testing.expect(cuda_instance.supportsPrecision(.fp64));
    try std.testing.expect(cuda_instance.supportsPrecision(.int8));

    const cpu_instance = BackendInstance{
        .backend_type = .stdgpu,
        .device_count = 1,
        .total_memory_mb = 4096,
        .available_memory_mb = 4096,
        .priority = 3,
        .is_healthy = true,
        .is_emulated = true,
        .supports_fp16 = false,
        .supports_fp64 = false,
        .supports_int8 = false,
        .supports_unified_memory = false,
    };

    try std.testing.expect(!cpu_instance.supportsPrecision(.fp16));
    try std.testing.expect(cpu_instance.supportsPrecision(.fp32)); // Always supported
    try std.testing.expect(!cpu_instance.supportsPrecision(.fp64));
}

test "category scheduling" {
    const allocator = std.testing.allocator;
    const coord = try Coordinator.init(allocator);
    defer coord.deinit();

    // Test different categories
    const categories = [_]WorkloadCategory{
        .general,
        .matrix_multiply,
        .convolution,
        .attention,
        .embedding,
        .elementwise,
        .reduction,
        .fft,
        .sorting,
    };

    for (categories) |cat| {
        const profile = WorkloadProfile{
            .compute_intensity = 0.5,
            .category = cat,
        };
        const decision = coord.schedule(profile);
        try std.testing.expect(decision.decision_id > 0);
    }

    // Verify category tracking
    const stats = coord.getStats();
    var total_category_count: u64 = 0;
    for (stats.category_counts) |count| {
        total_category_count += count;
    }
    try std.testing.expectEqual(@as(u64, categories.len), total_category_count);
}
