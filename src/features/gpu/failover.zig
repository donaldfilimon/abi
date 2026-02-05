//! Automatic GPU Backend Failover Manager
//!
//! Provides automatic failover between GPU backends when failures occur.
//! Works with recovery.zig to detect failures and trigger backend switches.
//!
//! ## Features
//! - Automatic backend switching on failure detection
//! - Health monitoring with configurable thresholds
//! - Failover priority ordering
//! - State preservation during failover
//!
//! ## Usage
//! ```zig
//! var failover = FailoverManager.init(allocator, .{
//!     .primary_backend = .cuda,
//!     .failover_chain = &.{ .vulkan, .stdgpu },
//! });
//! defer failover.deinit();
//!
//! // Register with GPU for automatic failover
//! gpu.setFailoverManager(&failover);
//! ```

const std = @import("std");
const backend_mod = @import("backend.zig");
const recovery = @import("recovery.zig");
const time = @import("../../services/shared/utils.zig");
const interface = @import("interface.zig");
const error_handling = @import("error_handling.zig");

pub const Backend = backend_mod.Backend;

/// Degradation mode for graceful fallback.
pub const DegradationMode = enum {
    /// No degradation - fail on GPU errors.
    none,
    /// Warn but continue with CPU fallback.
    warn_and_continue,
    /// Silent fallback to CPU.
    silent,
    /// Automatic - degrade if CPU fallback available.
    automatic,
};

/// Failover configuration.
pub const FailoverConfig = struct {
    /// Primary backend to use.
    primary_backend: Backend = .cuda,
    /// Ordered list of failover backends.
    failover_chain: []const Backend = &.{ .vulkan, .metal, .stdgpu },
    /// Maximum failures before triggering failover.
    failure_threshold: u32 = 3,
    /// Time window for counting failures (milliseconds).
    failure_window_ms: u64 = 30_000,
    /// Cooldown before retrying failed backend (milliseconds).
    backend_cooldown_ms: u64 = 60_000,
    /// Enable automatic recovery to primary backend.
    auto_recover_to_primary: bool = true,
    /// Interval for primary backend health check (milliseconds).
    primary_health_check_interval_ms: u64 = 30_000,
    /// Callback on failover events.
    on_failover: ?*const fn (FailoverEvent) void = null,
    /// Degradation mode for graceful fallback.
    degradation_mode: DegradationMode = .automatic,
    /// Always include CPU fallback as last resort.
    ensure_cpu_fallback: bool = true,
};

/// Failover event types.
pub const FailoverEvent = union(enum) {
    failover_started: struct {
        from: Backend,
        to: Backend,
        reason: FailoverReason,
    },
    failover_completed: struct {
        from: Backend,
        to: Backend,
        duration_ms: u64,
    },
    failover_failed: struct {
        from: Backend,
        to: Backend,
        @"error": FailoverError,
    },
    primary_restored: struct {
        backend: Backend,
    },
    all_backends_exhausted: void,
};

/// Reasons for failover.
pub const FailoverReason = enum {
    device_lost,
    repeated_failures,
    performance_degradation,
    memory_exhausted,
    manual_trigger,
};

/// Failover errors.
pub const FailoverError = error{
    NoAvailableBackend,
    BackendInitFailed,
    StateTransferFailed,
    AllBackendsExhausted,
    FailoverInProgress,
    CooldownActive,
};

/// Backend state tracking.
const BackendState = struct {
    status: Status,
    failure_count: u32,
    last_failure_time: i64,
    cooldown_until: i64,
    last_used_time: i64,

    const Status = enum {
        available,
        active,
        failed,
        cooldown,
        unavailable,
    };
};

/// Automatic failover manager.
pub const FailoverManager = struct {
    allocator: std.mem.Allocator,
    config: FailoverConfig,

    // State tracking
    current_backend: Backend,
    backend_states: std.AutoHashMapUnmanaged(Backend, BackendState),
    failover_history: std.ArrayListUnmanaged(FailoverHistoryEntry),

    // Synchronization
    mutex: std.Thread.Mutex,
    failover_in_progress: bool,

    // Integration with recovery
    recovery_manager: ?*recovery.RecoveryManager,

    const FailoverHistoryEntry = struct {
        timestamp: i64,
        from_backend: Backend,
        to_backend: Backend,
        reason: FailoverReason,
        success: bool,
        duration_ms: u64,
    };

    /// Initialize the failover manager.
    pub fn init(allocator: std.mem.Allocator, config: FailoverConfig) FailoverManager {
        var manager = FailoverManager{
            .allocator = allocator,
            .config = config,
            .current_backend = config.primary_backend,
            .backend_states = .empty,
            .failover_history = .empty,
            .mutex = .{},
            .failover_in_progress = false,
            .recovery_manager = null,
        };

        // Initialize primary backend state
        manager.backend_states.put(allocator, config.primary_backend, .{
            .status = .active,
            .failure_count = 0,
            .last_failure_time = 0,
            .cooldown_until = 0,
            .last_used_time = time.unixMilliseconds(),
        }) catch {
            std.debug.print("[gpu_failover] Failed to initialize primary backend state - failover may not work correctly\n", .{});
        };

        // Initialize failover chain states
        for (config.failover_chain) |backend| {
            manager.backend_states.put(allocator, backend, .{
                .status = .available,
                .failure_count = 0,
                .last_failure_time = 0,
                .cooldown_until = 0,
                .last_used_time = 0,
            }) catch {
                std.debug.print("[gpu_failover] Failed to initialize failover backend {t} state\n", .{backend});
            };
        }

        return manager;
    }

    /// Deinitialize the failover manager.
    pub fn deinit(self: *FailoverManager) void {
        self.backend_states.deinit(self.allocator);
        self.failover_history.deinit(self.allocator);
        self.* = undefined;
    }

    /// Connect to a recovery manager for coordinated handling.
    pub fn setRecoveryManager(self: *FailoverManager, rm: *recovery.RecoveryManager) void {
        self.recovery_manager = rm;
    }

    /// Get the current active backend.
    pub fn getCurrentBackend(self: *const FailoverManager) Backend {
        return self.current_backend;
    }

    /// Report a failure on the current backend.
    pub fn reportFailure(self: *FailoverManager, reason: FailoverReason) FailoverError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.unixMilliseconds();

        // Update failure count
        if (self.backend_states.getPtr(self.current_backend)) |state| {
            // Check if we're in a new failure window
            if (now - state.last_failure_time > @as(i64, @intCast(self.config.failure_window_ms))) {
                state.failure_count = 1;
            } else {
                state.failure_count += 1;
            }
            state.last_failure_time = now;

            // Check if threshold exceeded
            if (state.failure_count >= self.config.failure_threshold) {
                return self.triggerFailoverLocked(reason);
            }
        }
    }

    /// Manually trigger a failover.
    pub fn triggerFailover(self: *FailoverManager, reason: FailoverReason) FailoverError!void {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.triggerFailoverLocked(reason);
    }

    fn triggerFailoverLocked(self: *FailoverManager, reason: FailoverReason) FailoverError!void {
        if (self.failover_in_progress) {
            return FailoverError.FailoverInProgress;
        }

        self.failover_in_progress = true;
        defer self.failover_in_progress = false;

        const start_time = time.unixMilliseconds();
        const from_backend = self.current_backend;

        // Mark current backend as failed
        if (self.backend_states.getPtr(from_backend)) |state| {
            state.status = .failed;
            state.cooldown_until = start_time + @as(i64, @intCast(self.config.backend_cooldown_ms));
        }

        // Find next available backend
        const to_backend = self.findNextAvailableBackend() orelse {
            self.notifyEvent(.{ .all_backends_exhausted = {} });
            return FailoverError.AllBackendsExhausted;
        };

        // Notify start
        self.notifyEvent(.{
            .failover_started = .{
                .from = from_backend,
                .to = to_backend,
                .reason = reason,
            },
        });

        // Switch backend
        self.current_backend = to_backend;
        if (self.backend_states.getPtr(to_backend)) |state| {
            state.status = .active;
            state.last_used_time = start_time;
        }

        const duration = @as(u64, @intCast(time.unixMilliseconds() - start_time));

        // Record history
        self.failover_history.append(self.allocator, .{
            .timestamp = start_time,
            .from_backend = from_backend,
            .to_backend = to_backend,
            .reason = reason,
            .success = true,
            .duration_ms = duration,
        }) catch {
            std.debug.print("[gpu_failover] Failed to record failover history from {t} to {t}\n", .{ from_backend, to_backend });
        };

        // Notify completion
        self.notifyEvent(.{
            .failover_completed = .{
                .from = from_backend,
                .to = to_backend,
                .duration_ms = duration,
            },
        });
    }

    fn findNextAvailableBackend(self: *FailoverManager) ?Backend {
        const now = time.unixMilliseconds();

        // Try failover chain in order
        for (self.config.failover_chain) |backend| {
            if (self.backend_states.get(backend)) |state| {
                // Skip if in cooldown
                if (state.cooldown_until > now) {
                    continue;
                }
                // Skip if unavailable
                if (state.status == .unavailable) {
                    continue;
                }
                return backend;
            }
        }

        return null;
    }

    fn notifyEvent(self: *FailoverManager, event: FailoverEvent) void {
        if (self.config.on_failover) |callback| {
            callback(event);
        }
    }

    /// Check if primary backend can be restored.
    pub fn checkPrimaryRecovery(self: *FailoverManager) bool {
        if (!self.config.auto_recover_to_primary) {
            return false;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.current_backend == self.config.primary_backend) {
            return false; // Already on primary
        }

        const now = time.unixMilliseconds();
        if (self.backend_states.get(self.config.primary_backend)) |state| {
            // Check if cooldown expired
            if (state.cooldown_until <= now) {
                return true;
            }
        }

        return false;
    }

    /// Restore to primary backend.
    pub fn restorePrimary(self: *FailoverManager) FailoverError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const primary = self.config.primary_backend;

        // Mark current as available
        if (self.backend_states.getPtr(self.current_backend)) |state| {
            state.status = .available;
        }

        // Activate primary
        if (self.backend_states.getPtr(primary)) |state| {
            state.status = .active;
            state.failure_count = 0;
            state.last_used_time = time.unixMilliseconds();
        }

        self.current_backend = primary;
        self.notifyEvent(.{ .primary_restored = .{ .backend = primary } });
    }

    /// Get failover statistics.
    pub fn getStats(self: *const FailoverManager) FailoverStats {
        return .{
            .total_failovers = self.failover_history.items.len,
            .current_backend = self.current_backend,
            .is_on_primary = self.current_backend == self.config.primary_backend,
            .backends_exhausted = self.findNextAvailableBackend() == null,
            .is_degraded = self.current_backend == .stdgpu and self.config.primary_backend != .stdgpu,
            .degradation_mode = self.config.degradation_mode,
        };
    }

    // ========================================================================
    // Graceful Degradation
    // ========================================================================

    /// Check if graceful degradation to CPU is available.
    pub fn canDegradeToCpu(self: *const FailoverManager) bool {
        if (self.config.degradation_mode == .none) {
            return false;
        }

        // Check if stdgpu is in the failover chain or ensure_cpu_fallback is enabled
        if (self.config.ensure_cpu_fallback) {
            return true;
        }

        for (self.config.failover_chain) |backend| {
            if (backend == .stdgpu) {
                return true;
            }
        }

        return false;
    }

    /// Check if currently running in degraded mode (CPU fallback).
    pub fn isDegraded(self: *const FailoverManager) bool {
        return self.current_backend == .stdgpu and self.config.primary_backend != .stdgpu;
    }

    /// Gracefully degrade to CPU fallback.
    /// Returns true if degradation occurred, false if already on CPU or degradation disabled.
    pub fn degradeToCpu(self: *FailoverManager, reason: FailoverReason) FailoverError!bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Already on CPU fallback
        if (self.current_backend == .stdgpu) {
            return false;
        }

        // Check if degradation is allowed
        if (self.config.degradation_mode == .none) {
            return FailoverError.NoAvailableBackend;
        }

        // Force switch to stdgpu
        const from_backend = self.current_backend;
        const start_time = time.unixMilliseconds();

        // Mark current backend as failed
        if (self.backend_states.getPtr(from_backend)) |state| {
            state.status = .failed;
            state.cooldown_until = start_time + @as(i64, @intCast(self.config.backend_cooldown_ms));
        }

        // Ensure stdgpu is in backend_states
        if (self.backend_states.getPtr(.stdgpu) == null) {
            self.backend_states.put(self.allocator, .stdgpu, .{
                .status = .available,
                .failure_count = 0,
                .last_failure_time = 0,
                .cooldown_until = 0,
                .last_used_time = 0,
            }) catch return FailoverError.BackendInitFailed;
        }

        // Notify based on degradation mode
        if (self.config.degradation_mode == .warn_and_continue or
            self.config.degradation_mode == .automatic)
        {
            self.notifyEvent(.{
                .failover_started = .{
                    .from = from_backend,
                    .to = .stdgpu,
                    .reason = reason,
                },
            });
        }

        // Switch to CPU fallback
        self.current_backend = .stdgpu;
        if (self.backend_states.getPtr(.stdgpu)) |state| {
            state.status = .active;
            state.last_used_time = start_time;
        }

        const duration = @as(u64, @intCast(time.unixMilliseconds() - start_time));

        // Record history
        self.failover_history.append(self.allocator, .{
            .timestamp = start_time,
            .from_backend = from_backend,
            .to_backend = .stdgpu,
            .reason = reason,
            .success = true,
            .duration_ms = duration,
        }) catch {
            std.debug.print("[gpu_failover] Failed to record CPU fallback history from {t}\n", .{from_backend});
        };

        // Notify completion (unless silent mode)
        if (self.config.degradation_mode != .silent) {
            self.notifyEvent(.{
                .failover_completed = .{
                    .from = from_backend,
                    .to = .stdgpu,
                    .duration_ms = duration,
                },
            });
        }

        return true;
    }

    /// Attempt to upgrade from CPU fallback to a GPU backend.
    /// Useful for periodic retry of GPU backends after degradation.
    pub fn attemptUpgrade(self: *FailoverManager) FailoverError!bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Only attempt if currently on CPU fallback
        if (self.current_backend != .stdgpu) {
            return false;
        }

        const now = time.unixMilliseconds();

        // Try primary first if cooldown expired
        if (self.backend_states.get(self.config.primary_backend)) |state| {
            if (state.cooldown_until <= now and state.status != .unavailable) {
                return self.switchToBackendLocked(self.config.primary_backend);
            }
        }

        // Try other backends in failover chain (excluding stdgpu)
        for (self.config.failover_chain) |backend| {
            if (backend == .stdgpu) continue;
            if (self.backend_states.get(backend)) |state| {
                if (state.cooldown_until <= now and state.status != .unavailable) {
                    return self.switchToBackendLocked(backend);
                }
            }
        }

        return false;
    }

    fn switchToBackendLocked(self: *FailoverManager, to_backend: Backend) FailoverError!bool {
        const from_backend = self.current_backend;
        const start_time = time.unixMilliseconds();

        // Mark current as available
        if (self.backend_states.getPtr(from_backend)) |state| {
            state.status = .available;
        }

        // Activate new backend
        if (self.backend_states.getPtr(to_backend)) |state| {
            state.status = .active;
            state.failure_count = 0;
            state.last_used_time = start_time;
        }

        self.current_backend = to_backend;

        self.notifyEvent(.{
            .failover_completed = .{
                .from = from_backend,
                .to = to_backend,
                .duration_ms = 0,
            },
        });

        return true;
    }

    /// Mark a backend as permanently unavailable.
    pub fn markBackendUnavailable(self: *FailoverManager, backend: Backend) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.backend_states.getPtr(backend)) |state| {
            state.status = .unavailable;
        }
    }

    /// Mark a backend as available again.
    pub fn markBackendAvailable(self: *FailoverManager, backend: Backend) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.backend_states.getPtr(backend)) |state| {
            if (state.status == .unavailable) {
                state.status = .available;
                state.failure_count = 0;
                state.cooldown_until = 0;
            }
        }
    }
};

/// Failover statistics.
pub const FailoverStats = struct {
    total_failovers: usize,
    current_backend: Backend,
    is_on_primary: bool,
    backends_exhausted: bool,
    is_degraded: bool,
    degradation_mode: DegradationMode,
};

// ============================================================================
// Tests
// ============================================================================

test "FailoverManager initialization" {
    const allocator = std.testing.allocator;

    var manager = FailoverManager.init(allocator, .{
        .primary_backend = .cuda,
        .failover_chain = &.{ .vulkan, .stdgpu },
    });
    defer manager.deinit();

    try std.testing.expectEqual(Backend.cuda, manager.getCurrentBackend());
    try std.testing.expect(manager.getStats().is_on_primary);
}

test "FailoverManager failure threshold" {
    const allocator = std.testing.allocator;

    var manager = FailoverManager.init(allocator, .{
        .primary_backend = .cuda,
        .failover_chain = &.{ .vulkan, .stdgpu },
        .failure_threshold = 2,
    });
    defer manager.deinit();

    // First failure - no failover
    try manager.reportFailure(.repeated_failures);
    try std.testing.expectEqual(Backend.cuda, manager.getCurrentBackend());

    // Second failure - triggers failover
    try manager.reportFailure(.repeated_failures);
    try std.testing.expectEqual(Backend.vulkan, manager.getCurrentBackend());
}
