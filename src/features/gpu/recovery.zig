//! GPU device lost recovery and automatic fallback mechanisms.
//!
//! Handles device failures, provides automatic recovery strategies,
//! and manages graceful degradation to fallback backends.

const std = @import("std");
const platform_time = @import("../../services/shared/utils.zig");
const time = platform_time;
const sync = @import("../../services/shared/sync.zig");
const backend = @import("backend.zig");

/// Recovery strategy for device failures.
pub const RecoveryStrategy = enum {
    /// Retry the operation with exponential backoff.
    retry,
    /// Switch to a different device.
    switch_device,
    /// Fall back to CPU implementation.
    fallback_cpu,
    /// Fall back to software GPU simulation.
    fallback_simulated,
    /// Fail immediately without recovery.
    fail_fast,
};

/// Recovery configuration.
pub const RecoveryConfig = struct {
    /// Primary recovery strategy.
    strategy: RecoveryStrategy = .switch_device,
    /// Maximum retry attempts.
    max_retries: u32 = 3,
    /// Initial retry delay in milliseconds.
    initial_retry_delay_ms: u64 = 100,
    /// Maximum retry delay in milliseconds.
    max_retry_delay_ms: u64 = 5000,
    /// Enable automatic fallback.
    enable_fallback: bool = true,
    /// Callback for recovery events.
    on_recovery: ?*const fn (RecoveryEvent) void = null,
};

/// Recovery event types.
pub const RecoveryEvent = union(enum) {
    device_lost: struct {
        backend_type: backend.Backend,
        device_id: i32,
    },
    recovery_started: struct {
        strategy: RecoveryStrategy,
        attempt: u32,
    },
    recovery_succeeded: struct {
        strategy: RecoveryStrategy,
        attempts: u32,
    },
    recovery_failed: struct {
        strategy: RecoveryStrategy,
        attempts: u32,
    },
    fallback_activated: struct {
        from_backend: backend.Backend,
        to_backend: backend.Backend,
    },
};

/// Device health status.
pub const DeviceHealth = enum {
    healthy,
    degraded,
    unhealthy,
    lost,
};

/// Device health check result.
pub const HealthCheck = struct {
    status: DeviceHealth,
    error_count: u32,
    last_error_time: i64,
    consecutive_failures: u32,
    uptime_seconds: f64,
};

/// Recovery manager for handling device failures.
pub const RecoveryManager = struct {
    allocator: std.mem.Allocator,
    config: RecoveryConfig,
    device_health: std.AutoHashMapUnmanaged(DeviceKey, DeviceHealthState),
    recovery_history: std.ArrayListUnmanaged(RecoveryHistoryEntry),
    active_backend: backend.Backend,
    fallback_backends: std.ArrayListUnmanaged(backend.Backend),
    mutex: sync.Mutex,

    const DeviceKey = struct {
        backend_type: backend.Backend,
        device_id: i32,

        pub fn hash(self: DeviceKey) u64 {
            var hasher = std.hash.Wyhash.init(0);
            hasher.update(std.mem.asBytes(&self.backend_type));
            hasher.update(std.mem.asBytes(&self.device_id));
            return hasher.final();
        }

        pub fn eql(a: DeviceKey, b: DeviceKey) bool {
            return a.backend_type == b.backend_type and a.device_id == b.device_id;
        }
    };

    const DeviceHealthState = struct {
        status: DeviceHealth,
        error_count: u32,
        last_error_time: i64,
        consecutive_failures: u32,
        health_check_interval_ms: u64,
        last_health_check: i64,
        recovery_attempts: u32,
    };

    const RecoveryHistoryEntry = struct {
        timestamp: i64,
        device_key: DeviceKey,
        strategy: RecoveryStrategy,
        success: bool,
        attempts: u32,
    };

    /// Initialize the recovery manager.
    pub fn init(allocator: std.mem.Allocator, config: RecoveryConfig) RecoveryManager {
        return .{
            .allocator = allocator,
            .config = config,
            .device_health = .{},
            .recovery_history = .{},
            .active_backend = .cuda,
            .fallback_backends = .{},
            .mutex = .{},
        };
    }

    /// Deinitialize the recovery manager.
    pub fn deinit(self: *RecoveryManager) void {
        self.device_health.deinit(self.allocator);
        self.recovery_history.deinit(self.allocator);
        self.fallback_backends.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register a device for health monitoring.
    pub fn registerDevice(
        self: *RecoveryManager,
        backend_type: backend.Backend,
        device_id: i32,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.registerDeviceUnlocked(backend_type, device_id);
    }

    /// Internal: Register device without acquiring mutex (caller must hold lock).
    fn registerDeviceUnlocked(
        self: *RecoveryManager,
        backend_type: backend.Backend,
        device_id: i32,
    ) !void {
        const key = DeviceKey{
            .backend_type = backend_type,
            .device_id = device_id,
        };

        const state = DeviceHealthState{
            .status = .healthy,
            .error_count = 0,
            .last_error_time = 0,
            .consecutive_failures = 0,
            .health_check_interval_ms = 5000,
            .last_health_check = time.nowSeconds(),
            .recovery_attempts = 0,
        };

        try self.device_health.put(self.allocator, key, state);
    }

    /// Report a device error.
    pub fn reportError(
        self: *RecoveryManager,
        backend_type: backend.Backend,
        device_id: i32,
        error_type: ErrorType,
    ) !RecoveryResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = DeviceKey{
            .backend_type = backend_type,
            .device_id = device_id,
        };

        const state_ptr = self.device_health.getPtr(key) orelse {
            // Use unlocked version since we already hold the mutex
            try self.registerDeviceUnlocked(backend_type, device_id);
            return .{ .action = .none };
        };

        state_ptr.error_count += 1;
        state_ptr.last_error_time = time.nowSeconds();
        state_ptr.consecutive_failures += 1;

        // Update health status based on error type and history
        if (error_type == .device_lost) {
            state_ptr.status = .lost;
        } else if (state_ptr.consecutive_failures >= 5) {
            state_ptr.status = .unhealthy;
        } else if (state_ptr.consecutive_failures >= 3) {
            state_ptr.status = .degraded;
        }

        // Determine recovery action
        if (state_ptr.status == .lost) {
            return self.initiateRecovery(key, state_ptr);
        }

        return .{ .action = .none };
    }

    /// Attempt to recover from device failure.
    pub fn recover(
        self: *RecoveryManager,
        backend_type: backend.Backend,
        device_id: i32,
    ) !RecoveryResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = DeviceKey{
            .backend_type = backend_type,
            .device_id = device_id,
        };

        const state_ptr = self.device_health.getPtr(key) orelse {
            return error.DeviceNotRegistered;
        };

        return self.initiateRecovery(key, state_ptr);
    }

    /// Check device health.
    pub fn checkHealth(
        self: *RecoveryManager,
        backend_type: backend.Backend,
        device_id: i32,
    ) ?HealthCheck {
        self.mutex.lock();
        defer self.mutex.unlock();

        const key = DeviceKey{
            .backend_type = backend_type,
            .device_id = device_id,
        };

        const state = self.device_health.get(key) orelse return null;

        const uptime = @as(f64, @floatFromInt(time.nowSeconds() - state.last_health_check));

        return .{
            .status = state.status,
            .error_count = state.error_count,
            .last_error_time = state.last_error_time,
            .consecutive_failures = state.consecutive_failures,
            .uptime_seconds = uptime,
        };
    }

    /// Get recovery history.
    pub fn getHistory(self: *const RecoveryManager) []const RecoveryHistoryEntry {
        return self.recovery_history.items;
    }

    /// Clear recovery history.
    pub fn clearHistory(self: *RecoveryManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.recovery_history.clearRetainingCapacity();
    }

    // Internal methods
    fn initiateRecovery(
        self: *RecoveryManager,
        key: DeviceKey,
        state: *DeviceHealthState,
    ) RecoveryResult {
        const strategy = self.config.strategy;

        // Emit recovery started event
        if (self.config.on_recovery) |callback| {
            callback(.{ .recovery_started = .{
                .strategy = strategy,
                .attempt = state.recovery_attempts + 1,
            } });
        }

        state.recovery_attempts += 1;

        const result = switch (strategy) {
            .retry => self.executeRetryStrategy(key, state),
            .switch_device => self.executeSwitchDeviceStrategy(key, state),
            .fallback_cpu => self.executeFallbackStrategy(.stdgpu, key, state),
            .fallback_simulated => self.executeFallbackStrategy(.simulated, key, state),
            .fail_fast => RecoveryResult{ .action = .fail },
        };

        // Record in history
        self.recordRecovery(key, strategy, result.action != .fail, state.recovery_attempts) catch {
            std.debug.print("[gpu_recovery] Failed to record recovery history for key: {d}\n", .{key});
        };

        // Emit recovery completed event
        if (self.config.on_recovery) |callback| {
            if (result.action != .fail) {
                callback(.{ .recovery_succeeded = .{
                    .strategy = strategy,
                    .attempts = state.recovery_attempts,
                } });
            } else {
                callback(.{ .recovery_failed = .{
                    .strategy = strategy,
                    .attempts = state.recovery_attempts,
                } });
            }
        }

        return result;
    }

    fn executeRetryStrategy(
        self: *RecoveryManager,
        key: DeviceKey,
        state: *DeviceHealthState,
    ) RecoveryResult {
        _ = key;
        if (state.recovery_attempts >= self.config.max_retries) {
            return .{ .action = .fail };
        }

        // Calculate exponential backoff delay
        const attempt = state.recovery_attempts;
        const delay_ms = @min(
            self.config.initial_retry_delay_ms * (@as(u64, 1) << @intCast(attempt)),
            self.config.max_retry_delay_ms,
        );

        return .{
            .action = .retry,
            .retry_delay_ms = delay_ms,
        };
    }

    fn executeSwitchDeviceStrategy(
        self: *RecoveryManager,
        key: DeviceKey,
        state: *DeviceHealthState,
    ) RecoveryResult {
        _ = self;
        _ = state;
        _ = key;
        // Find alternative device in the same backend
        // For now, just return a switch action
        return .{
            .action = .switch_device,
            .new_device_id = 0, // Would be determined by device discovery
        };
    }

    fn executeFallbackStrategy(
        self: *RecoveryManager,
        fallback_backend: backend.Backend,
        key: DeviceKey,
        state: *DeviceHealthState,
    ) RecoveryResult {
        _ = state;

        // Emit fallback event
        if (self.config.on_recovery) |callback| {
            callback(.{ .fallback_activated = .{
                .from_backend = key.backend_type,
                .to_backend = fallback_backend,
            } });
        }

        self.active_backend = fallback_backend;

        return .{
            .action = .fallback,
            .fallback_backend = fallback_backend,
        };
    }

    fn recordRecovery(
        self: *RecoveryManager,
        key: DeviceKey,
        strategy: RecoveryStrategy,
        success: bool,
        attempts: u32,
    ) !void {
        const entry = RecoveryHistoryEntry{
            .timestamp = time.unixSeconds(),
            .device_key = key,
            .strategy = strategy,
            .success = success,
            .attempts = attempts,
        };

        try self.recovery_history.append(self.allocator, entry);

        // Keep history limited to last 100 entries
        if (self.recovery_history.items.len > 100) {
            _ = self.recovery_history.orderedRemove(0);
        }
    }
};

/// Error types that can trigger recovery.
pub const ErrorType = enum {
    device_lost,
    out_of_memory,
    timeout,
    initialization_failed,
    synchronization_failed,
    other,
};

/// Recovery action result.
pub const RecoveryResult = struct {
    action: Action,
    retry_delay_ms: u64 = 0,
    new_device_id: i32 = 0,
    fallback_backend: backend.Backend = .simulated,

    pub const Action = enum {
        none,
        retry,
        switch_device,
        fallback,
        fail,
    };
};

test "recovery manager registration" {
    const allocator = std.testing.allocator;
    var manager = RecoveryManager.init(allocator, .{});
    defer manager.deinit();

    try manager.registerDevice(.cuda, 0);

    const health = manager.checkHealth(.cuda, 0).?;
    try std.testing.expectEqual(DeviceHealth.healthy, health.status);
}

test "error reporting and recovery" {
    const allocator = std.testing.allocator;
    var manager = RecoveryManager.init(allocator, .{
        .strategy = .retry,
        .max_retries = 3,
    });
    defer manager.deinit();

    try manager.registerDevice(.cuda, 0);

    // Report device lost error
    const result = try manager.reportError(.cuda, 0, .device_lost);
    try std.testing.expectEqual(RecoveryResult.Action.retry, result.action);

    const health = manager.checkHealth(.cuda, 0).?;
    try std.testing.expectEqual(DeviceHealth.lost, health.status);
}

test "fallback strategy" {
    const allocator = std.testing.allocator;
    var manager = RecoveryManager.init(allocator, .{
        .strategy = .fallback_simulated,
    });
    defer manager.deinit();

    try manager.registerDevice(.cuda, 0);

    const result = try manager.recover(.cuda, 0);
    try std.testing.expectEqual(RecoveryResult.Action.fallback, result.action);
    try std.testing.expectEqual(backend.Backend.simulated, result.fallback_backend);
}
